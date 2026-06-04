import torch
import torch.nn as nn
import absl.flags as flags
import numpy as np

FLAGS = flags.FLAGS

from tools.rot_utils import rotation_matrix_to_quaternion
from tools.umeyama_utils import umeyama
from network.fs_net_repo.FaceRecon import FaceRecon
from losses.f_loss import *
from network.fs_net_repo.gmm import *
from network.fs_net_repo.PoseR import *
from network.fs_net_repo.PoseTs import Pose_Ts

channels = [16,32,48,64,80,96,112]
voxel_size = (1 / 100, 1 / 100, 1 / 100)
device = 'cuda'

def weight_init(shape, mode, fan_in, fan_out):
    if mode == 'xavier_uniform': return np.sqrt(6 / (fan_in + fan_out)) * (torch.rand(*shape) * 2 - 1)
    if mode == 'xavier_normal':  return np.sqrt(2 / (fan_in + fan_out)) * torch.randn(*shape)
    if mode == 'kaiming_uniform': return np.sqrt(3 / fan_in) * (torch.rand(*shape) * 2 - 1)
    if mode == 'kaiming_normal':  return np.sqrt(1 / fan_in) * torch.randn(*shape)
    raise ValueError(f'Invalid init mode "{mode}"')

class SAFAG(nn.Module):

    def __init__(self, gapart):
        super(SAFAG, self).__init__()
        self.gapart = gapart
        self.gapart_name = ['Hinge_Knob', 'Hinge_Door', 'Slider_Button', 'Hinge_Knob', 'Line_Fixed_Handle',
                            'Round_Fixed_Handle', 'Slider_Drawer', 'Slider_Lid', 'Hinge_Lid', 'Hinge_Handle']
        self.gapart_name2id = {'Line_Fixed_Handle': 1, 'Round_Fixed_Handle': 2, 'Slider_Button': 3, 'Hinge_Door': 4,
                               'Slider_Drawer': 5, 'Slider_Lid': 6, 'Hinge_Lid': 7, 'Hinge_Knob': 8,
                               'Hinge_Handle': 9}
        self.symmetry_correspondence = {1: 'mirror', 2: 'C_n', 3: 'C_n', 4: 'mirror', 5: 'none', 6: 'C_n',
                                        7: 'mirror', 8: 'C_n', 9: 'mirror'}
        if self.gapart in self.gapart_name:
            self.gapart_id = self.gapart_name2id[self.gapart]
        self.symmetry = self.symmetry_check(self.gapart_id)
        self.backbone = FaceRecon()
        self.candidates_generator = ConditionalQuaternionSampler()
        self.encoder = CandidateDistributionEncoder()

        self.fusion = FeatureFusion()
        self.rot = final_quaternion()
        self.candidates_revision = AnchorLevelPredictor()
        self.ts = Pose_Ts()
        self.translation_loss = translation_loss()

        self.simplified_loss = simplified_joint_loss()
        self.candidates_loss = candidates_loss()

        self.sym_axis_gmm = SymAxisGMM().to(device)
        self.symmetry_loss = SymmetryAwareLoss().to(device)
        self.candidates_loss_axis = candidates_loss_axis()
        self.candidates_loss_axis_mix = CandidatesLossAxisMixture()

        self.sym_axis_gmm_mirror = AdaptiveSymmetryPredictor().to(device)
        self.symmetry_loss_mirror = SymMirrorAwareLoss().to(device)
        self.candidates_loss_mirror = candidates_loss_normal()
        self.candidates_loss_mirror_mix = CandidatesLossMirrorMixture()

        self.refine_head = QuaternionRefineHead(feat_dim=1290, quat_dim=4)
        self.pose_embed = PoseEmbedding(in_dim=4, embed_dim=64)

    def forward(self, pts,gt_R, gt_t, npcs,gt_s, obj_id,epoch,do_loss=False,sym=None):

        output_dict = {}

        gt_trans_umeyama = torch.zeros((len(pts),3))
        gt_rot_umeyama = torch.zeros((len(pts), 3,3))

        for i in range(len(pts)):
            gapart = pts[i].detach().cpu().numpy()
            npcs_pts = npcs[i].detach().cpu().numpy()
            c,r,t = umeyama(npcs_pts.transpose(),gapart.transpose())
            gt_rot_umeyama[i] = torch.tensor(r)
            gt_trans_umeyama[i] = torch.tensor(t).squeeze()
        gt_trans_umeyama = gt_trans_umeyama.to(device)
        gt_rot_umeyama = gt_rot_umeyama.to(device)

        recon, face, feat = self.backbone(pts - pts.mean(dim=1, keepdim=True), obj_id)

        if isinstance(sym, torch.Tensor):
            sym_flat = sym.view(-1)

            if not torch.all(sym_flat == sym_flat[0]):
                raise ValueError(
                    f"Inconsistent sym_info values within the current batch: "
                    f"{sym_flat.detach().cpu().tolist()}"
                )

            sym = int(sym_flat[0].item())
        else:
            sym = int(sym)

        if epoch <= FLAGS.warm_up_epoch:
            quaternion_candidates = self.candidates_generator(feat.permute(0, 2, 1))
            gt_quaternion = rotation_matrix_to_quaternion(gt_R)
            output_dict['candidates'] = quaternion_candidates
            output_dict['gt_quaternion'] = gt_quaternion
        else:
            quaternion_candidates = self.candidates_generator(feat.permute(0, 2, 1))

            encoding = self.encoder(quaternion_candidates)
            fused_feat = self.fusion(feat, encoding)

            feat_for_ts = torch.cat([feat, pts - pts.mean(dim=1, keepdim=True)], dim=2)
            T, s = self.ts(feat_for_ts.permute(0, 2, 1))
            final_quaternion = self.candidates_revision(fused_feat.permute(0, 2, 1), quaternion_candidates)


            global_feat = feat.mean(dim=1)

            coarse_q = final_quaternion

            pose_emb = self.pose_embed(coarse_q)


            delta_q = self.refine_head(
                feat=global_feat,
                coarse_q=coarse_q,
                pose_emb=pose_emb
            )

            refined_q = coarse_q + delta_q

            refined_q = refined_q / refined_q.norm(dim=-1, keepdim=True)

            p_Q = refined_q
            p_T = T + pts.mean(dim=1)
            p_s = s

            gt_quaternion = rotation_matrix_to_quaternion(gt_rot_umeyama)
            output_dict['recon'] = recon
            output_dict['PC'] = pts
            output_dict['Pred_Q'] = p_Q
            output_dict['Pred_T'] = p_T
            output_dict['Pred_s'] = s
            output_dict['gt_R'] = gt_rot_umeyama
            output_dict['gt_t'] = gt_trans_umeyama
            output_dict['gt_s'] = gt_s
            output_dict['gt_quaternion'] = gt_quaternion
            output_dict['candidates'] = quaternion_candidates

            if (sym == 1):
                axis, angle = quat_to_axis_angle(p_Q.detach())  # [B,3], [B,1]
                rot_feat = torch.cat([axis, angle], dim=-1)  # [B,4]

                weighted_axis, pi_probs,  axes,log_probs = self.sym_axis_gmm(rot_feat, feat)

                output_dict['weighted_axis'] =weighted_axis
                output_dict['pi_probs'] = pi_probs

            if (sym == 2):
                axis, angle = quat_to_axis_angle(p_Q.detach())  # [B,3], [B,1]
                rot_feat = torch.cat([axis, angle], dim=-1)  # [B,4]


                sym_normals, geom_weights,  pi_probs = self.sym_axis_gmm_mirror(rot_feat, feat, pts,gt_rot_umeyama)

                output_dict['sym_normals'] =sym_normals
                output_dict['geom_weights'] = geom_weights
                output_dict['pi_probs'] = pi_probs

        if do_loss:
            if (sym == 0):
                recon_loss = FLAGS.recon_w * nn.L1Loss()(recon, pts)
                if epoch > FLAGS.warm_up_epoch:
                    cal_loss_dict = self.simplified_loss(gt_quaternion, gt_trans_umeyama, p_Q, p_T)
                    candidates_loss, candidates_loss_dict = self.candidates_loss(quaternion_candidates, gt_quaternion)
                    loss_dict = {}
                    loss_dict['rot_loss'] = FLAGS.rot_w * cal_loss_dict['rot_loss']
                    loss_dict['trans_loss'] = FLAGS.trans_w * cal_loss_dict['trans_loss']
                    loss_dict['candidates_loss'] = 2 * candidates_loss
                    loss_dict['recon_loss'] = recon_loss
                else:
                    loss_dict = {}
                    candidates_loss, cal_loss_dict = self.candidates_loss(quaternion_candidates, gt_quaternion)
                    loss_dict['candidates_loss'] = 5 * candidates_loss
                    loss_dict['recon_loss'] = recon_loss

            if (sym == 1):
                recon_loss = FLAGS.recon_w * nn.L1Loss()(recon, pts)
                if epoch > FLAGS.warm_up_epoch:
                    trans_loss = self.translation_loss(gt_trans_umeyama, p_T)
                    rot_loss = self.symmetry_loss(
                        pred_q=p_Q,
                        gt_q=gt_quaternion,
                        weighted_axis=weighted_axis,
                        pi_probs=pi_probs,
                        axes=axes
                    )
                    candidates_loss = self.candidates_loss_axis(
                        q_cands=quaternion_candidates,
                        q_gt=gt_quaternion,
                        sym_axis=weighted_axis
                    )
                    loss_dict = {
                        'rot_loss': FLAGS.rot_w * rot_loss,
                        'trans_loss': FLAGS.trans_w * trans_loss,
                        'candidates_loss': 2 * candidates_loss,
                        'recon_loss':recon_loss
                    }
                else:
                    loss_dict = {}
                    axes_xyz = torch.eye(3, device=device, dtype=gt_quaternion.dtype)
                    candidates_loss = self.candidates_loss_axis_mix(
                        q_cands=quaternion_candidates,
                        q_gt=gt_quaternion,
                        axes=axes_xyz
                    )
                    loss_dict['candidates_loss'] = 5 * candidates_loss
                    loss_dict['recon_loss'] = recon_loss

            if (sym == 2):
                recon_loss = FLAGS.recon_w * nn.L1Loss()(recon, pts)
                if epoch > FLAGS.warm_up_epoch:
                    trans_loss = self.translation_loss(gt_trans_umeyama, p_T)
                    rot_loss = self.symmetry_loss_mirror(
                        pred_q=p_Q,
                        gt_q=gt_quaternion,
                        gt_R=gt_rot_umeyama,
                        sym_normals=sym_normals,
                        pts=pts,
                        geom_weights=geom_weights
                    )
                    candidates_loss = self.candidates_loss_mirror(
                        q_cands=quaternion_candidates,
                        q_gt=gt_quaternion,
                        gt_R=gt_rot_umeyama,
                        sym_normals=sym_normals,
                        pts=pts,
                        geom_weights=geom_weights
                    )
                    loss_dict = {
                        'rot_loss': FLAGS.rot_w * rot_loss,
                        'trans_loss': FLAGS.trans_w * trans_loss,
                        'candidates_loss': 2 * candidates_loss,
                        'recon_loss': recon_loss
                    }
                else:
                    loss_dict = {}
                    axes_xyz = torch.eye(3, device=device, dtype=gt_quaternion.dtype)  # [3,3]
                    candidates_loss = self.candidates_loss_mirror_mix(
                        q_cands=quaternion_candidates,
                        q_gt=gt_quaternion,
                        gt_R=gt_rot_umeyama,
                        axes=axes_xyz,
                        pts=pts
                    )
                    loss_dict['candidates_loss'] = 5 * candidates_loss
                    loss_dict['recon_loss'] = recon_loss
        else:
            return output_dict

        return output_dict,loss_dict

    def symmetry_check(self,obj_id):
        symmetry = self.symmetry_correspondence[obj_id]
        return symmetry


