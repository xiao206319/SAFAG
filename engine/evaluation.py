import os
import math
import itertools
import time

from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

from absl import app
from absl import flags

from config.config import *
from network.SAFAGPose_test import SAFAG
from datasets.load_data_test import PoseDataset
from tools.vis_utils import *


FLAGS = flags.FLAGS

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    if not torch.is_tensor(R1):
        R1 = torch.tensor(R1, dtype=torch.float32)
    if not torch.is_tensor(R2):
        R2 = torch.tensor(R2, dtype=torch.float32)
    if not torch.is_tensor(sym_axis):
        sym_axis = torch.tensor(sym_axis, dtype=torch.float32)

    device_, dtype = R1.device, R1.dtype

    R1 = R1.to(device=device_, dtype=dtype)
    R2 = R2.to(device=device_, dtype=dtype)
    sym_axis = sym_axis.to(device=device_, dtype=dtype)

    sym_axis = F.normalize(sym_axis, dim=0, eps=eps)

    v1 = F.normalize(R1 @ sym_axis, dim=0, eps=eps)
    v2 = F.normalize(R2 @ sym_axis, dim=0, eps=eps)

    cosang = torch.abs(torch.dot(v1, v2)).clamp(-1.0, 1.0)
    theta = torch.acos(cosang) * 180.0 / torch.pi

    return theta.item() if torch.is_tensor(theta) else theta


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
    if not torch.is_tensor(R1):
        R1 = torch.tensor(R1, dtype=torch.float32)
    if not torch.is_tensor(R2):
        R2 = torch.tensor(R2, dtype=torch.float32)
    if not torch.is_tensor(normals):
        normals = torch.tensor(normals, dtype=torch.float32)

    device_, dtype = R1.device, R1.dtype

    R1 = R1.to(device=device_, dtype=dtype)
    R2 = R2.to(device=device_, dtype=dtype)
    normals = normals.to(device=device_, dtype=dtype)
    normals = F.normalize(normals, dim=-1, eps=eps)

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


def rot_error(r_gt, r_pred):
    R1 = r_gt / np.cbrt(np.linalg.det(r_gt))
    R2 = r_pred / np.cbrt(np.linalg.det(r_gt))

    R = R1 @ R2.transpose()
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)
    theta *= 180 / np.pi

    return theta


def parse_batch_sym(sym_info):
    if isinstance(sym_info, torch.Tensor):
        sym_flat = sym_info.view(-1)

        if not torch.all(sym_flat == sym_flat[0]):
            raise ValueError(
                f"Inconsistent sym_info values within the current batch: "
                f"{sym_flat.detach().cpu().tolist()}"
            )

        return int(sym_flat[0].item())

    return int(sym_info)


def load_checkpoint_to_network(network, ckpt_path):
    if ckpt_path == '':
        raise ValueError(
            "FLAGS.test_model is empty. Please set test_model in config.py."
        )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint

    network.load_state_dict(state_dict)
    print(f"[Loaded checkpoint] {ckpt_path}")


def test_one_split(network, dataset, split_name, eval_epoch):
    test_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=FLAGS.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=True
    )

    progress_bar_test = tqdm(enumerate(test_dataloader), total=len(test_dataloader))

    print(f'test_{split_name} !!!')

    total_angle_diff_per_epoch = 0.0
    total_translation_diff_per_epoch = 0.0

    best_angle_diff = 10000.0
    best_translation_diff = 10000.0

    count_10deg_10cm = 0
    count_5deg_5cm = 0
    count_5deg_2cm = 0
    total_valid = 0

    with torch.no_grad():
        for i, data in progress_bar_test:
            output_dict_val = network(
                pts=data['pts'].to(device),
                npcs=data['npcs'].to(device) if torch.is_tensor(data['npcs']) else data['npcs'],
                gt_R=data['rotation'].to(device),
                gt_t=data['translation'].to(device),
                epoch=eval_epoch,
                gt_s=data['scale'].to(device),
                obj_id=data['id'].to(device),
                do_loss=False,
                sym=data['sym_info']
            )

            pred_quarternion_list = output_dict_val['Pred_Q']
            pred_trans_list = output_dict_val['Pred_T']
            pred_rot_list = quaternion_to_rotation_matrix(pred_quarternion_list)

            gt_rot_list = output_dict_val['gt_R'].to(device)
            gt_trans_list = output_dict_val['gt_t'].to(device)

            sym = parse_batch_sym(data['sym_info'])

            if sym == 2:
                n1, n2, n3 = output_dict_val['sym_normals']
                sym_normals = torch.stack((n1, n2, n3), dim=1).to(device)

            assert len(gt_rot_list) == len(gt_trans_list), 'data loading failed'

            total_angle_diff = 0.0
            total_translation_diff = 0.0

            nan_count = 0
            nan_count_angle = 0

            for j in range(len(pred_rot_list)):
                pred_rotation = pred_rot_list[j, :, :].detach().cpu().numpy()
                pred_translation = pred_trans_list[j, :].detach().cpu().numpy()

                gt_rotation = gt_rot_list[j, :, :].detach().cpu().numpy()
                gt_translation = gt_trans_list[j, :].detach().cpu().numpy()

                if sym == 0:
                    angle_diff = rot_error(pred_rotation, gt_rotation)

                elif sym == 1:
                    angle_diff = rot_error_axis_symmetric(
                        torch.tensor(pred_rotation, device=device, dtype=torch.float32),
                        torch.tensor(gt_rotation, device=device, dtype=torch.float32),
                        output_dict_val['weighted_axis'][j, :]
                    )

                elif sym == 2:
                    selected = sym_normals[j, :].detach().cpu().numpy()
                    angle_diff = mirror_normal_error_multi(
                        gt_rotation,
                        pred_rotation,
                        selected
                    )

                else:
                    raise ValueError(f"Unknown sym_info value: {sym}")

                if not math.isnan(angle_diff):
                    translation_diff = np.linalg.norm(gt_translation - pred_translation)
                    total_valid += 1

                    if angle_diff <= 10 and translation_diff <= 0.10:
                        count_10deg_10cm += 1
                    if angle_diff <= 5 and translation_diff <= 0.05:
                        count_5deg_5cm += 1
                    if angle_diff <= 5 and translation_diff <= 0.02:
                        count_5deg_2cm += 1

                if np.any(np.isnan(np.abs(pred_translation - gt_translation))):
                    nan_count += 1
                    translation_diff = 0.0
                else:
                    translation_diff = np.linalg.norm(gt_translation - pred_translation)

                total_translation_diff += translation_diff

                if math.isnan(angle_diff):
                    nan_count_angle += 1
                else:
                    total_angle_diff += angle_diff

            valid_rot_num = len(gt_rot_list) - nan_count_angle
            valid_trans_num = len(gt_rot_list) - nan_count

            average_rot_diff = total_angle_diff / valid_rot_num if valid_rot_num > 0 else 0.0
            average_trans_diff = total_translation_diff / valid_trans_num if valid_trans_num > 0 else 0.0

            if average_rot_diff < best_angle_diff:
                best_angle_diff = average_rot_diff

            if average_trans_diff < best_translation_diff:
                best_translation_diff = average_trans_diff

            total_angle_diff_per_epoch += average_rot_diff
            total_translation_diff_per_epoch += average_trans_diff

            progress_bar_test.set_description(f"testing {split_name} batch {i + 1}")

    if total_valid > 0:
        acc_10deg_10cm = 100 * count_10deg_10cm / total_valid
        acc_5deg_5cm = 100 * count_5deg_5cm / total_valid
        acc_5deg_2cm = 100 * count_5deg_2cm / total_valid
    else:
        acc_10deg_10cm = acc_5deg_5cm = acc_5deg_2cm = 0.0

    average_rot_diff_per_batch = total_angle_diff_per_epoch / len(test_dataloader)
    average_trans_diff_per_batch = total_translation_diff_per_epoch / len(test_dataloader)


    return {
        f'{split_name}_average_rot_diff': average_rot_diff_per_batch,
        f'{split_name}_average_trans_diff': average_trans_diff_per_batch,
        f'{split_name}_best_rot_diff': best_angle_diff,
        f'{split_name}_best_translation_diff': best_translation_diff,
        f'{split_name}_acc_10deg_10cm': acc_10deg_10cm,
        f'{split_name}_acc_5deg_5cm': acc_5deg_5cm,
        f'{split_name}_acc_5deg_2cm': acc_5deg_2cm,
    }


def test(argv):
    if not os.path.exists(FLAGS.model_save):
        os.makedirs(FLAGS.model_save)

    log_path = os.path.join(FLAGS.model_save, FLAGS.gapart)
    os.makedirs(log_path, exist_ok=True)


    eval_epoch = FLAGS.warm_up_epoch + 1
    network = SAFAG(gapart=FLAGS.gapart)
    network = network.to(device)

    load_checkpoint_to_network(network, FLAGS.test_model)
    network.eval()

    val_dataset_intra = PoseDataset(
        mode='test',
        test_mode='intra',
        per_obj=FLAGS.gapart,
        n_pts=FLAGS.n_points
    )

    val_dataset_inter = PoseDataset(
        mode='test',
        test_mode='inter',
        per_obj=FLAGS.gapart,
        n_pts=FLAGS.n_points
    )

    intra_result = test_one_split(
        network=network,
        dataset=val_dataset_intra,
        split_name='intra',
        eval_epoch=eval_epoch
    )

    inter_result = test_one_split(
        network=network,
        dataset=val_dataset_inter,
        split_name='inter',
        eval_epoch=eval_epoch
    )



    print('==================================================')
    print('Final Summary')
    print(f"intra_rot_diff: {intra_result['intra_average_rot_diff']:.4f}")
    print(f"inter_rot_diff: {inter_result['inter_average_rot_diff']:.4f}")
    print(f"intra_trans_diff: {intra_result['intra_average_trans_diff']:.4f}")
    print(f"inter_trans_diff: {inter_result['inter_average_trans_diff']:.4f}")
    print(f"intra_acc_10deg_10cm: {intra_result['intra_acc_10deg_10cm']:.2f}%")
    print(f"inter_acc_10deg_10cm: {inter_result['inter_acc_10deg_10cm']:.2f}%")
    print(f"intra_acc_5deg_5cm: {intra_result['intra_acc_5deg_5cm']:.2f}%")
    print(f"inter_acc_5deg_5cm: {inter_result['inter_acc_5deg_5cm']:.2f}%")
    print(f"intra_acc_5deg_2cm: {intra_result['intra_acc_5deg_2cm']:.2f}%")
    print(f"inter_acc_5deg_2cm: {inter_result['inter_acc_5deg_2cm']:.2f}%")
    print('==================================================')



if __name__ == "__main__":
    app.run(test)