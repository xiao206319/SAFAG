import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
from absl import app

import time
import random
from absl import flags
import itertools

from tools.training_utils import build_lr_rate
from network.SAFAGPose_test import SAFAG
from tools.vis_utils import *

FLAGS = flags.FLAGS
from datasets.load_data_test import PoseDataset

import tensorflow as tf
from tools.eval_utils import setup_logger
device = 'cuda'

EPS = 1e-8

def quaternion_to_rotation_matrix(quat):
    quat = F.normalize(quat, dim=-1)

    w, x, y, z = quat.unbind(dim=-1)

    B = quat.shape[0]

    rot_mat = torch.stack([
        1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)
    ], dim=-1).reshape(B, 3, 3)

    return rot_mat

def rot_error_axis_symmetric(R1, R2, sym_axis, eps=1e-8):
    if not torch.is_tensor(R1): R1 = torch.tensor(R1)
    if not torch.is_tensor(R2): R2 = torch.tensor(R2)
    if not torch.is_tensor(sym_axis): sym_axis = torch.tensor(sym_axis)

    device, dtype = R1.device, R1.dtype
    R1, R2 = R1.to(device, dtype), R2.to(device, dtype)
    sym_axis = F.normalize(sym_axis.to(device, dtype), dim=0, eps=eps)

    v1 = F.normalize(R1 @ sym_axis, dim=0, eps=eps)
    v2 = F.normalize(R2 @ sym_axis, dim=0, eps=eps)

    cosang = torch.abs(torch.dot(v1, v2)).clamp(-1.0, 1.0)
    theta = torch.acos(cosang) * 180.0 / torch.pi
    return theta

def householder_from_normal(n: torch.Tensor) -> torch.Tensor:

    n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
    I = torch.eye(3, device=n.device, dtype=n.dtype).expand(*n.shape[:-1], 3, 3)
    nnT = n.unsqueeze(-1) @ n.unsqueeze(-2)
    return I - 2.0 * nnT

def generate_equiv_poses_single(R_gt: torch.Tensor, normals: list) -> torch.Tensor:

    M = len(normals)
    S_list = [householder_from_normal(n) for n in normals]
    I3 = torch.eye(3, device=R_gt.device, dtype=R_gt.dtype)

    combos = list(itertools.product([0, 1], repeat=M))
    R_equivs = []
    for combo in combos:
        S_combo = I3.clone()
        for k, flip in enumerate(combo):
            if flip:
                S_combo = S_list[k] @ S_combo
        R_equivs.append(R_gt @ S_combo)

    return torch.stack(R_equivs, dim=0)


def mirror_normal_error_multi(R1, R2, normals, eps=1e-8):

    if not torch.is_tensor(R1): R1 = torch.tensor(R1, dtype=torch.float32)
    if not torch.is_tensor(R2): R2 = torch.tensor(R2, dtype=torch.float32)
    if not torch.is_tensor(normals): normals = torch.tensor(normals, dtype=torch.float32)

    device, dtype = R1.device, R1.dtype
    R1 = R1.to(device=device, dtype=dtype)
    R2 = R2.to(device=device, dtype=dtype)
    normals = F.normalize(normals.to(device=device, dtype=dtype), dim=-1, eps=eps)

    R_equivs = generate_equiv_poses_single(R1, normals)

    errs = []
    for R_eq in R_equivs:
        M = R2.T @ R_eq
        trace = torch.trace(M)
        cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        angle = torch.acos(cos_theta)
        errs.append(angle)

    errs = torch.stack(errs)
    min_err = errs.min()
    return (min_err * 180.0 / torch.pi).item()



def rot_error(r_gt,r_pred):
    R1 = r_gt / np.cbrt(np.linalg.det(r_gt))
    R2 = r_pred / np.cbrt(np.linalg.det(r_gt))

    R = R1 @ R2.transpose()
    theta = np.arccos((np.trace(R) - 1) / 2)
    theta *= 180 / np.pi
    return theta



def train(argv):

    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)
    tf.compat.v1.disable_eager_execution()
    tb_writter = tf.compat.v1.summary.FileWriter(FLAGS.model_save)

    logger = setup_logger('train_log', os.path.join(FLAGS.model_save, 'log.txt'))
    logger_save = setup_logger('save_log', os.path.join(FLAGS.model_save, f'val_log_{FLAGS.gapart}_save.txt'))
    logger_test = setup_logger('val_log',os.path.join(FLAGS.model_save, f'val_log_{FLAGS.gapart}_new.txt'))
    logger_loss = setup_logger('loss_log', os.path.join(FLAGS.model_save, f'loss_log_{FLAGS.gapart}_new.txt'))

    network = SAFAG(gapart=FLAGS.gapart)
    network = network.to(device)
    if FLAGS.resume:
        network.load_state_dict(torch.load(FLAGS.resume_model))
        s_epoch = FLAGS.resume_point
        print(f'loading model from {FLAGS.resume_model},starting in epoch {FLAGS.resume_point}')
    else:
        s_epoch = 0

    train_dataset = PoseDataset(mode='train', per_obj=FLAGS.gapart,n_pts=1024)
    val_dataset = PoseDataset(mode='test',test_mode='intra',per_obj=FLAGS.gapart,n_pts=1024)
    val_dataset_inter = PoseDataset(mode='test', test_mode='inter', per_obj=FLAGS.gapart, n_pts=1024)

    st_time = time.time()
    train_steps = FLAGS.train_steps
    global_step = train_steps * s_epoch
    train_size = train_steps * FLAGS.batch_size
    indices = []
    page_start = - train_size

    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=1e-4,
        betas=(0.5, 0.999),
        eps=1e-6,
        weight_decay=0
    )
    optimizer.zero_grad()
    scheduler = build_lr_rate(optimizer, total_iters=train_steps * FLAGS.total_epoch // FLAGS.accumulate)
    best_intra_rot_diff = float('inf')
    best_inter_rot_diff = float('inf')
    best_all_rot_diff = float('inf')

    for epoch in range(s_epoch+1,FLAGS.total_epoch+1):
        logger.info(f'gapart {FLAGS.gapart} training !!!')
        logger.info('Time {0}'.format(time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - st_time)) + \
                                      ', ' + 'Epoch %02d' % epoch + ', ' + 'Training started'))


        page_start += train_size
        len_last = len(indices) - page_start
        if len_last < train_size:
            indices = indices[page_start:]
            data_list = list(range(train_dataset.length))
            try:
                for i in range((train_size - len_last) // train_dataset.length + 1):
                    random.shuffle(data_list)
                    indices += data_list
            except:
                import ipdb
                ipdb.set_trace()
            page_start = 0

        train_idx = indices[page_start:(page_start + train_size)]
        train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size,
                                                       sampler=train_sampler, drop_last=True,
                                                       num_workers=FLAGS.num_workers, pin_memory=True)


        network.train()
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        print("ready to train")

        for i,data in progress_bar:
            output_dict,fsnet_loss = network(pts=data['pts'].to(device),npcs=data["npcs"],gt_R=data['rotation'].to(device), gt_t=data['translation'].to(device),epoch=epoch,
                                       gt_s=data['scale'].to(device), obj_id=data['id'].to(device),do_loss=True,sym=data['sym_info'])
            total_loss = sum(fsnet_loss.values())

            if epoch > FLAGS.warm_up_epoch:
                if i % 100 == 0:
                    logger_loss.info(
                        f"[Epoch {epoch:02d}] [Batch {i + 1:03d}]\n "
                        f"rot_loss = {fsnet_loss['rot_loss']:.4f} \n"
                        f"candidates_loss = {fsnet_loss['candidates_loss']:.4f} \n"
                        f"recon_loss = {fsnet_loss['recon_loss']:.4f} \n"
                    )
            else:
                if i % 100 == 0:
                    logger_loss.info(
                        f"[Epoch {epoch:02d}] [Batch {i + 1:03d}]\n "
                        f"candidates_loss = {fsnet_loss['candidates_loss']:.4f} \n"
                        f"recon_loss = {fsnet_loss['recon_loss']:.4f} \n"

                    )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), 5)

            if (global_step + 1) % FLAGS.accumulate == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.set_description(f"Processing batch {i + 1}")

        network.eval()
        logger_test.info(f'epoch_{epoch} is validating now')
        test_idx = np.random.choice(len(val_dataset), size=192, replace=False)
        test_sampler = torch.utils.data.sampler.SubsetRandomSampler(test_idx)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=FLAGS.batch_size,
                                                       sampler=test_sampler,
                                                       num_workers=0, pin_memory=True)
        val_dataloader_inter = torch.utils.data.DataLoader(val_dataset_inter, batch_size=FLAGS.batch_size,
                                                     sampler=test_sampler,
                                                     num_workers=0, pin_memory=True)
        progress_bar_test = tqdm(enumerate(val_dataloader), total=len(val_dataloader))
        progress_bar_test_inter = tqdm(enumerate(val_dataloader_inter), total=len(val_dataloader_inter))

        with torch.no_grad():

            print('test_intra !!!')
            total_angle_diff_per_epoch = 0
            total_translation_diff_per_epoch = 0
            best_angle_diff = 10000
            best_translation_diff = 10000

            count_10deg_10cm = 0
            count_5deg_5cm = 0
            count_5deg_2cm = 0
            total_valid = 0


            for i,data in progress_bar_test:
                output_dict_val = network(pts=data['pts'].to(device),npcs=data["npcs"], gt_R=data['rotation'].to(device),
                                            gt_t=data['translation'].to(device),epoch=epoch,
                                            gt_s=data['scale'].to(device), obj_id=data['id'].to(device), do_loss=False,
                                            sym=data['sym_info'])
                if epoch <= FLAGS.warm_up_epoch:
                    quaternion_candidates = output_dict_val['candidates']
                    gt_quaternion = output_dict_val['gt_quaternion']
                    assert gt_quaternion.shape[0] == quaternion_candidates.shape[0],'data failed !'
                    bs = gt_quaternion.shape[0]
                    sampling_idx = random.randint(0,bs-1)
                    root = os.path.join(f'/home/chenwenxiao/IJCAI-2026/visualization/{FLAGS.gapart}/intra',str(epoch))
                    os.makedirs(root, exist_ok=True)
                    filename = 'visu_'+str(sampling_idx)+'.png'
                    path = os.path.join(root,filename)
                    visualize_candidates_on_sphere(quaternion_candidates[sampling_idx],gt_quaternion[sampling_idx],path)
                    continue

                pred_quarternion_list = output_dict_val['Pred_Q']
                pred_trans_list = output_dict_val['Pred_T']
                pred_rot_list = quaternion_to_rotation_matrix(pred_quarternion_list)
                gt_rot_list = output_dict_val['gt_R'].to(device)
                gt_trans_list = output_dict_val['gt_t'].to(device)

                if isinstance(data['sym_info'], torch.Tensor):
                    sym_flat = data['sym_info'].view(-1)

                    if not torch.all(sym_flat == sym_flat[0]):
                        raise ValueError(
                            f"Inconsistent sym_info values within the current batch: "
                            f"{sym_flat.detach().cpu().tolist()}"
                        )

                    sym = int(sym_flat[0].item())
                else:
                    sym = int(data['sym_info'])


                if sym == 2:
                    n1, n2, n3 = output_dict_val['sym_normals']
                    sym_normals = torch.stack((n1, n2, n3), dim=1).to(device)

                assert len(gt_rot_list)==len(gt_trans_list),'data loading failed'

                total_angle_diff = 0
                total_translation_diff = 0
                nan_count = 0
                nan_count_angle = 0

                for j in range(len(pred_rot_list)):
                    translation_diff = 0
                    pred_rotation = pred_rot_list[j,:,:].cpu().numpy()
                    pred_translation = pred_trans_list[j,:].cpu().numpy()
                    gt_rotation = gt_rot_list[j,:,:].cpu().numpy()
                    gt_translation = gt_trans_list[j,:].cpu().numpy()

                    if(sym== 0):
                        angle_diff = rot_error(pred_rotation, gt_rotation)

                    if (sym == 1):
                        angle_diff = rot_error_axis_symmetric(pred_rotation, gt_rotation,
                                                              output_dict_val['weighted_axis'][j, :])
                    if (sym == 2):
                        selected = sym_normals[j, :].cpu().numpy()
                        angle_diff = mirror_normal_error_multi(gt_rotation, pred_rotation, selected)

                    if not math.isnan(angle_diff):
                        translation_diff = np.linalg.norm(gt_translation - pred_translation)
                        total_valid += 1

                        if angle_diff <= 10 and translation_diff <= 0.10:
                            count_10deg_10cm += 1
                        if angle_diff <= 5 and translation_diff <= 0.05:
                            count_5deg_5cm += 1
                        if angle_diff <= 5 and translation_diff <= 0.02:
                            count_5deg_2cm += 1


                    if(np.any(np.isnan(np.abs(pred_translation-gt_translation)))):
                        nan_count = nan_count + 1
                    else:
                        translation_diff = np.linalg.norm(gt_translation - pred_translation)

                    total_translation_diff = total_translation_diff + translation_diff

                    if (math.isnan(angle_diff)):
                        nan_count_angle = nan_count_angle + 1
                    else:
                        total_angle_diff = total_angle_diff + angle_diff

                average_rot_diff = total_angle_diff / (len(gt_rot_list)-nan_count_angle)
                average_trans_diff = total_translation_diff / (len(gt_rot_list)-nan_count)


                if(average_rot_diff<best_angle_diff):
                    best_angle_diff = average_rot_diff
                if(average_trans_diff<best_translation_diff):
                    best_translation_diff = average_trans_diff

                total_angle_diff_per_epoch = total_angle_diff_per_epoch + average_rot_diff
                total_translation_diff_per_epoch = total_translation_diff_per_epoch + average_trans_diff

                progress_bar_test.set_description(f"testing batch {i + 1}")

            if total_valid > 0:
                acc_10deg_10cm = 100 * count_10deg_10cm / total_valid
                acc_5deg_5cm = 100 * count_5deg_5cm / total_valid
                acc_5deg_2cm = 100 * count_5deg_2cm / total_valid
            else:
                acc_10deg_10cm = acc_5deg_5cm = acc_5deg_2cm = 0.0

            average_rot_diff_per_batch = total_angle_diff_per_epoch / len(val_dataloader)
            average_trans_diff_per_batch = total_translation_diff_per_epoch / len(val_dataloader)
            logger_test.info(f'test_intra diff:\n')
            logger_test.info(f'epoch {epoch} \n'
                             f'average_rot_diff:{average_rot_diff_per_batch:.4f} \n'
                             f'average_trans_diff:{average_trans_diff_per_batch:.4f} \n'
                             f'best_rot_diff:{best_angle_diff:.4f} \n'
                             f'best_translation_diff:{best_translation_diff:.4f} \n'
                             f'Acc(10° 10cm): {acc_10deg_10cm:.2f}% \n'
                             f'Acc(5° 5cm): {acc_5deg_5cm:.2f}% \n'
                             f'Acc(5° 2cm): {acc_5deg_2cm:.2f}% \n')

            if average_rot_diff_per_batch!=0:
                average_rot_diff_per_batch_intra=average_rot_diff_per_batch

            print('test_inter !!!')

            total_angle_diff_per_epoch = 0
            total_translation_diff_per_epoch = 0
            best_angle_diff = 10000
            best_translation_diff = 10000

            count_10deg_10cm = 0
            count_5deg_5cm = 0
            count_5deg_2cm = 0
            total_valid = 0

            for i, data in progress_bar_test_inter:
                output_dict_val = network(pts=data['pts'].to(device), gt_R=data['rotation'].to(device),
                                          npcs=data['npcs'].to(device),
                                          gt_t=data['translation'].to(device), epoch=epoch,
                                          gt_s=data['scale'].to(device), obj_id=data['id'].to(device), do_loss=False,
                                          sym=data['sym_info'])
                if epoch <= FLAGS.warm_up_epoch:
                    quaternion_candidates = output_dict_val['candidates']
                    gt_quaternion = output_dict_val['gt_quaternion']
                    assert gt_quaternion.shape[0] == quaternion_candidates.shape[0],'data failed !'
                    bs = gt_quaternion.shape[0]
                    sampling_idx = random.randint(0,bs-1)
                    root = os.path.join(f'/home/chenwenxiao/IJCAI-2026/visualization/{FLAGS.gapart}/inter',str(epoch))
                    os.makedirs(root, exist_ok=True)
                    filename = 'visu_'+str(sampling_idx)+'.png'
                    path = os.path.join(root,filename)
                    visualize_candidates_on_sphere(quaternion_candidates[sampling_idx],gt_quaternion[sampling_idx],path)
                    continue

                pred_quarternion_list = output_dict_val['Pred_Q']
                pred_trans_list = output_dict_val['Pred_T']
                pred_rot_list = quaternion_to_rotation_matrix(pred_quarternion_list)
                gt_rot_list = output_dict_val['gt_R'].to(device)
                gt_trans_list = output_dict_val['gt_t'].to(device)

                if sym == 2:
                    n1, n2, n3 = output_dict_val['sym_normals']
                    sym_normals = torch.stack((n1, n2, n3), dim=1).to(device)

                assert len(gt_rot_list)==len(gt_trans_list),'data loading failed'

                total_angle_diff = 0
                total_translation_diff = 0
                nan_count = 0
                nan_count_angle = 0

                for j in range(len(pred_rot_list)):
                    translation_diff = 0
                    pred_rotation = pred_rot_list[j,:,:].cpu().numpy()
                    pred_translation = pred_trans_list[j,:].cpu().numpy()
                    gt_rotation = gt_rot_list[j,:,:].cpu().numpy()
                    gt_translation = gt_trans_list[j,:].cpu().numpy()

                    if(data['sym_info']== 0):
                        angle_diff = rot_error(pred_rotation, gt_rotation)

                    if (data['sym_info'] == 1):
                        angle_diff = rot_error_axis_symmetric(pred_rotation, gt_rotation,
                                                              output_dict_val['weighted_axis'][j, :])
                    if (sym == 2):
                        selected = sym_normals[j, :].cpu().numpy()
                        angle_diff = mirror_normal_error_multi(gt_rotation, pred_rotation, selected)

                    if not math.isnan(angle_diff):
                        translation_diff = np.linalg.norm(gt_translation - pred_translation)
                        total_valid += 1

                        if angle_diff <= 10 and translation_diff <= 0.10:
                            count_10deg_10cm += 1
                        if angle_diff <= 5 and translation_diff <= 0.05:
                            count_5deg_5cm += 1
                        if angle_diff <= 5 and translation_diff <= 0.02:
                            count_5deg_2cm += 1

                    if (np.any(np.isnan(np.abs(pred_translation - gt_translation)))):
                        nan_count = nan_count + 1
                    else:
                        translation_diff = np.linalg.norm(gt_translation - pred_translation)
                    total_translation_diff = total_translation_diff + translation_diff
                    if( math.isnan(angle_diff)):
                        nan_count_angle = nan_count_angle + 1
                    else:
                        total_angle_diff = total_angle_diff + angle_diff

                average_rot_diff = total_angle_diff / (len(gt_rot_list) - nan_count_angle)
                average_trans_diff = total_translation_diff / (len(gt_rot_list) - nan_count)

                if (average_rot_diff < best_angle_diff):
                    best_angle_diff = average_rot_diff
                if (average_trans_diff < best_translation_diff):
                    best_translation_diff = average_trans_diff


                total_angle_diff_per_epoch = total_angle_diff_per_epoch + average_rot_diff
                total_translation_diff_per_epoch = total_translation_diff_per_epoch + average_trans_diff

                progress_bar_test.set_description(f"testing batch {i + 1}")

            if total_valid > 0:
                acc_10deg_10cm = 100 * count_10deg_10cm / total_valid
                acc_5deg_5cm = 100 * count_5deg_5cm / total_valid
                acc_5deg_2cm = 100 * count_5deg_2cm / total_valid
            else:
                acc_10deg_10cm = acc_5deg_5cm = acc_5deg_2cm = 0.0



            average_rot_diff_per_batch = total_angle_diff_per_epoch / len(val_dataloader)
            average_trans_diff_per_batch = total_translation_diff_per_epoch / len(val_dataloader)

            logger_test.info(f'test_inter diff:\n')
            logger_test.info(f'epoch {epoch} \n'
                             f'average_rot_diff:{average_rot_diff_per_batch:.4f} \n'
                             f'average_trans_diff:{average_trans_diff_per_batch:.4f} \n'
                             f'best_rot_diff:{best_angle_diff:.4f} \n'
                             f'best_translation_diff:{best_translation_diff:.4f} \n'
                             f'Acc(10° 10cm): {acc_10deg_10cm:.2f}% \n'
                             f'Acc(5° 5cm): {acc_5deg_5cm:.2f}% \n'
                             f'Acc(5° 2cm): {acc_5deg_2cm:.2f}% \n')
            if average_rot_diff_per_batch != 0:
                average_rot_diff_per_batch_inter=average_rot_diff_per_batch

        logger.info('>>>>>>>>----------Epoch {:02d} train finish---------<<<<<<<<\n'.format(epoch))

        save_dir = os.path.join(FLAGS.model_save, FLAGS.gapart)
        os.makedirs(save_dir, exist_ok=True)

        ckpt_path = os.path.join(save_dir, f'{epoch}.pth')
        torch.save(network.state_dict(), ckpt_path)

        logger_save.info(
            f'[CKPT SAVED][ALL] Epoch {epoch} | '
        )

if __name__ == "__main__":
    app.run(train)

