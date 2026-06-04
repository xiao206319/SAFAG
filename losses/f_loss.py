import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import itertools

EPS = 1e-8

def safe_normalize(x, dim=-1, eps=1e-8):
    norm = torch.norm(x, p=2, dim=dim, keepdim=True)
    return x / (norm + eps)

def quat_normalize(q, eps=EPS):
    return q / (q.norm(dim=-1, keepdim=True) + eps)

def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def quat_angle(q1, q2):

    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = torch.sum(q1 * q2, dim=-1)
    dot = torch.abs(dot)
    dot = torch.clamp(dot, -1.0 + 1e-7, 1.0 - 1e-7)
    return 2.0 * torch.acos(dot)


def quaternion_to_rotation_matrix(quat):

    assert quat.shape[-1] == 4, "quat must end with 4 elements"

    quat = F.normalize(quat, dim=-1)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    rot_mat = torch.stack([
        1 - 2 * (y*y + z*z),   2 * (x*y - z*w),     2 * (x*z + y*w),
        2 * (x*y + z*w),       1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w),       2 * (y*z + x*w),     1 - 2 * (x*x + y*y),
    ], dim=-1)

    out_shape = quat.shape[:-1] + (3,3)
    rot_mat = rot_mat.reshape(out_shape)

    return rot_mat


def axis_angle_to_quat(axis, angle):

    device, dtype = axis.device, axis.dtype

    if not torch.is_tensor(angle):
        angle = torch.tensor(angle, device=device, dtype=dtype)
    else:
        angle = angle.to(device=device, dtype=dtype)

    if angle.dim() == 0:
        angle = angle.expand(axis.size(0))
    elif angle.dim() == 2 and angle.size(-1) == 1:
        angle = angle.squeeze(-1)

    axis = F.normalize(axis, dim=-1)
    half = angle / 2
    cos_half = torch.cos(half).unsqueeze(-1)
    sin_half = torch.sin(half).unsqueeze(-1)

    q = torch.cat([cos_half, sin_half * axis], dim=-1)
    return q


def householder_from_normal(n: torch.Tensor) -> torch.Tensor:

    n = n / (n.norm(dim=-1, keepdim=True) + 1e-8)
    I = torch.eye(3, device=n.device, dtype=n.dtype).expand(*n.shape[:-1], 3, 3)
    nnT = n.unsqueeze(-1) @ n.unsqueeze(-2)
    return I - 2.0 * nnT

def generate_equiv_poses(R_gt: torch.Tensor, normals: list) -> torch.Tensor:

    if R_gt.dim() == 2:
        R_gt = R_gt.unsqueeze(0)
    B = R_gt.size(0)
    M = len(normals)

    S_list = [householder_from_normal(n) for n in normals]
    I3 = torch.eye(3, device=R_gt.device, dtype=R_gt.dtype).unsqueeze(0).expand(B,3,3)

    combos = list(itertools.product([0,1], repeat=M))
    S_combos = []
    for combo in combos:
        S = I3.clone()
        for k, flip in enumerate(combo):
            if flip:
                S = S_list[k] @ S
        S_combos.append(S)

    R_equivs = [R_gt @ S for S in S_combos]
    R_equivs = torch.stack(R_equivs, dim=1)
    return R_equivs


class candidates_loss_axis(nn.Module):
    def __init__(self, num_samples=36,m_target=0.85, min_sep_deg=0.5):
        super(candidates_loss_axis, self).__init__()
        self.num_samples = num_samples
        self.m_target = m_target
        self.min_sep = math.radians(min_sep_deg)
        self.angles = torch.linspace(0, 2 * torch.pi, steps=num_samples)  # [N]

    def forward(self, q_cands, q_gt, sym_axis):
        """
        q_cands: [B,K,4]   候选四元数
        q_gt:    [B,4]     GT 四元数
        sym_axis:[B,3]     预测的对称轴 (单位化)
        """
        B, K, _ = q_cands.shape
        q_cands = F.normalize(q_cands, dim=-1)
        q_gt = F.normalize(q_gt, dim=-1)

        # --- angles 跟随 sym_axis 的 device/dtype ---
        angles = self.angles.to(sym_axis.device, sym_axis.dtype)

        # 构造等价解集 (GT沿对称轴旋转多个角度)
        equivs = []
        for ang in angles:
            q_rot_pos = axis_angle_to_quat(sym_axis, ang)  # [B,4]
            q_rot_neg = axis_angle_to_quat(-sym_axis, ang)  # [B,4]
            q_equiv_pos = quat_mul(q_gt, q_rot_pos)
            q_equiv_neg = quat_mul(q_gt, q_rot_neg)

            # 双方向取最小误差（无方向对称）
            q_equiv = torch.stack([q_equiv_pos, q_equiv_neg], dim=1)  # [B,2,4]
            equivs.append(q_equiv)

        equivs = torch.stack(equivs, dim=1).view(B, -1, 4)  # [B, 2*num_angles, 4]

        # 计算每个 candidate 和所有等价解的误差
        loss_per_cand = []
        for k in range(K):
            cand = q_cands[:, k, :]                        # [B,4]
            ang_errs = quat_angle(
                cand.unsqueeze(1).expand(-1, equivs.size(1), -1),  # [B,N,4]
                equivs                                             # [B,N,4]
            )  # [B,N]
            min_err, _ = torch.min(ang_errs, dim=-1)  # [B]
            loss_per_cand.append(min_err)

        loss_per_cand = torch.stack(loss_per_cand, dim=1)  # [B,K]
        align_loss = loss_per_cand.mean()                  # scalar

        total_loss = align_loss

        return total_loss


#用于warmup阶段
class CandidatesLossAxisMixture(nn.Module):
    def __init__(self, num_samples=36, beta=10,m_target=0.85, min_sep_deg=0.5):
        super().__init__()
        self.register_buffer("angles", torch.linspace(0, 2 * torch.pi, steps=num_samples))
        self.num_samples = num_samples
        self.beta = beta
        self.m_target = m_target
        self.min_sep = math.radians(min_sep_deg)

    def softmin(self, x):
        w = F.softmax(-self.beta * x, dim=-1)
        return (w * x).sum(dim=-1)

    def forward(self, q_cands, q_gt, axes):
        B, K, _ = q_cands.shape
        q_cands = F.normalize(q_cands, dim=-1)
        q_gt = F.normalize(q_gt, dim=-1)

        device, dtype = q_gt.device, q_gt.dtype
        angles = self.angles.to(device=device, dtype=dtype)

        if axes.dim() == 2:  # [3,3] -> [B,3,3]
            axes = axes.unsqueeze(0).expand(B, -1, -1)
        elif axes is None:
            axes = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)

        axes = F.normalize(axes, dim=-1)

        per_axis_loss = []

        # -------- 对每个轴采样旋转等价解 --------
        for k in range(3):
            axis = axes[:, k, :]  # [B,3]
            q_equiv_list = []

            for ang in angles:
                q_rot_pos = axis_angle_to_quat(axis, ang)  # [B,4]
                q_rot_neg = axis_angle_to_quat(-axis, ang)  # [B,4]
                q_eq_pos = quat_mul(q_gt, q_rot_pos)
                q_eq_neg = quat_mul(q_gt, q_rot_neg)
                q_equiv_list.extend([q_eq_pos, q_eq_neg])

            q_equiv = torch.stack(q_equiv_list, dim=1)  # [B, 2N, 4]
            q_equiv = F.normalize(q_equiv, dim=-1)

            # ---------- 比较所有候选 ----------
            qc = q_cands.unsqueeze(2).expand(B, K, q_equiv.size(1), 4)
            qe = q_equiv.unsqueeze(1).expand(B, K, q_equiv.size(1), 4)
            err = quat_angle(qc, qe)  # [B,K,2N]
            min_over_angle, _ = torch.min(err, dim=-1)  # [B,K]
            min_over_cand= torch.mean(min_over_angle, dim=-1)  # [B]
            per_axis_loss.append(min_over_cand)

        loss = torch.stack(per_axis_loss, dim=-1)
        align_loss = loss.min(dim=-1)[0].mean()

        total_loss = align_loss

        return total_loss



class CandidatesLossMirrorMixture(nn.Module):

    def __init__(self, mirror_weight=2.0, eps=1e-6):
        super().__init__()
        self.mirror_weight = mirror_weight
        self.eps = eps

    def mirror_consistency_loss(self, points, n_axes):

        B, N, _ = points.shape
        n_axes = F.normalize(n_axes, dim=-1)
        c = points.mean(dim=1, keepdim=True)
        loss_per_axis = []

        for k in range(3):
            n = n_axes[:, k, :].unsqueeze(1)
            proj = torch.sum((points - c) * n, dim=-1, keepdim=True)
            mirror_points = points - 2 * proj * n
            dist = torch.cdist(points, mirror_points, p=2)
            min_dist1, _ = torch.min(dist, dim=2)
            min_dist2, _ = torch.min(dist, dim=1)
            loss_k = (min_dist1.mean(dim=1) + min_dist2.mean(dim=1)).mean()
            loss_per_axis.append(loss_k)

        loss_per_axis = torch.stack(loss_per_axis, dim=-1)
        min_loss, _ = torch.min(loss_per_axis, dim=-1)
        return min_loss / 3.0

    def forward(self, q_cands, q_gt, gt_R,axes, pts=None):
        B, K, _ = q_cands.shape
        device, dtype = q_cands.device, q_cands.dtype

        q_cands = F.normalize(q_cands, dim=-1)

        if axes.dim() == 2:
            axes = axes.unsqueeze(0).expand(B, -1, -1)
        axes = F.normalize(axes, dim=-1)

        normals_list = [axes[:, 0, :], axes[:, 1, :], axes[:, 2, :]]
        R_equivs = generate_equiv_poses(gt_R, normals_list)
        B, N_eq = R_equivs.shape[:2]
        B, K, _ = q_cands.shape
        q_cands_flat = F.normalize(q_cands.reshape(B * K, 4), dim=-1)
        R_cands_flat = quaternion_to_rotation_matrix(q_cands_flat)
        R_cands = R_cands_flat.reshape(B, K, 3, 3)

        errs_all = []
        for i in range(N_eq):
            R_eq_i = R_equivs[:, i, :, :]

            R_eq_i_expand = R_eq_i.unsqueeze(1).expand(B, K, 3, 3)
            R_pred = R_cands
            R_rel = torch.matmul(R_pred.transpose(-1, -2), R_eq_i_expand)

            trace = R_rel[:, :, 0, 0] + R_rel[:, :, 1, 1] + R_rel[:, :, 2, 2]
            cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            angle_rad = torch.acos(cos_theta)
            errs_all.append(angle_rad)

        errs_all = torch.stack(errs_all, dim=-1)
        min_err_per_cand, _ = errs_all.min(dim=-1)
        loss_angle = min_err_per_cand.mean()

        axes_obj = torch.bmm(gt_R.transpose(1, 2), axes.transpose(1, 2)).transpose(1, 2)
        axes_obj = F.normalize(axes_obj, dim=-1)


        if pts is not None:
            loss_geom = self.mirror_consistency_loss(pts, axes_obj)
            total_loss =  loss_angle + 2 * loss_geom
        else:
            total_loss = loss_angle

        return total_loss

class candidates_loss_normal(nn.Module):
    def __init__(self, mirror_weight=2.0, eps=1e-8):
        super().__init__()
        self.mirror_weight = mirror_weight
        self.eps = eps

    @staticmethod
    def mirror_consistency_loss(points, n_pred):
        B, N, _ = points.shape
        n_pred = F.normalize(n_pred, dim=-1)
        c = points.mean(dim=1, keepdim=True)
        n = n_pred.unsqueeze(1)
        proj = torch.sum((points - c) * n, dim=-1, keepdim=True)
        mirror_points = points - 2 * proj * n
        dist = torch.cdist(points, mirror_points, p=2)
        min_dist1, _ = torch.min(dist, dim=2)
        min_dist2, _ = torch.min(dist, dim=1)
        return (min_dist1.mean(dim=1) + min_dist2.mean(dim=1)) / 2

    def forward(self, q_cands, q_gt, gt_R,sym_normals, pts=None, geom_weights=None):

        B, K, _ = q_cands.shape
        q_cands = F.normalize(q_cands, dim=-1)

        n1, n2, n3 = sym_normals
        n1 = F.normalize(n1, dim=-1)
        n2 = F.normalize(n2, dim=-1)
        n3 = F.normalize(n3, dim=-1)

        n1_obj = F.normalize(
            torch.bmm(gt_R.transpose(1, 2), n1.unsqueeze(-1)).squeeze(-1),
            dim=-1, eps=1e-8
        )
        n2_obj = F.normalize(
            torch.bmm(gt_R.transpose(1, 2), n2.unsqueeze(-1)).squeeze(-1),
            dim=-1, eps=1e-8
        )
        n3_obj = F.normalize(
            torch.bmm(gt_R.transpose(1, 2), n3.unsqueeze(-1)).squeeze(-1),
            dim=-1, eps=1e-8
        )

        if pts is not None:
            loss_n1 = self.mirror_consistency_loss(pts, n1_obj)
            loss_n2 = self.mirror_consistency_loss(pts, n2_obj)
            loss_n3 = self.mirror_consistency_loss(pts, n3_obj)


        mean_w = geom_weights.mean(dim=0)
        w1, w2, w3 = mean_w[0], mean_w[1], mean_w[2]

        if w1 > 0.6 and (w1 - w2) > 0.2:
            sym_plane_count = 1
        elif (w1 + w2) > 0.85 and (w2 - w3) > 0.1:
            sym_plane_count = 2
        else:
            sym_plane_count = 3

        if sym_plane_count == 1:
            normals_used = [n1]
        elif sym_plane_count == 2:
            normals_used = [n1, n2]
        else:
            normals_used = [n1, n2, n3]

        R_equivs = generate_equiv_poses(gt_R, normals_used)

        B, K, _ = q_cands.shape
        q_cands_flat = F.normalize(q_cands.reshape(B * K, 4), dim=-1)
        R_cands_flat = quaternion_to_rotation_matrix(q_cands_flat)
        R_cands = R_cands_flat.reshape(B, K, 3, 3)

        B, K, N_eq = R_cands.shape[0], R_cands.shape[1], R_equivs.shape[1]

        errs_all = torch.zeros((B, K, N_eq), device=q_cands.device)
        for i in range(N_eq):
            R_eq_i = R_equivs[:, i, :, :]
            R_eq_i_exp = R_eq_i.unsqueeze(1).expand(-1, K, -1, -1)
            R_pred_T = R_cands.transpose(-1, -2)

            M = torch.matmul(R_pred_T, R_eq_i_exp)
            trace = M[:, :, 0, 0] + M[:, :, 1, 1] + M[:, :, 2, 2]
            cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            angle_deg = torch.acos(cos_theta)
            errs_all[:, :, i] = angle_deg

        min_err_per_cand = errs_all.min(dim=-1)[0]
        loss_angle = min_err_per_cand.mean()

        if sym_plane_count == 1:
            geom_loss = loss_n1.mean()
        elif sym_plane_count == 2:
            geom_loss = ((loss_n1 + loss_n2) / 2).mean()
        else:
            geom_loss = ((loss_n1 + loss_n2 + loss_n3) / 3).mean()
        total_loss =  2 * loss_angle +  geom_loss

        return total_loss



class translation_loss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.t_loss = nn.L1Loss()

    def trans_loss(self, pred_translation, gt_translation):
        return self.t_loss(pred_translation, gt_translation)

    def forward(self, gt_translation, pred_translation):
        trans_loss = self.trans_loss(pred_translation, gt_translation)

        return trans_loss

def loss_stage1_none(q_cands, q_gt):
    """
    Asymmetric: align to GT, control concentration via mean(dot^2),
    small repulsion to avoid total collapse.
    """
    B, K, _ = q_cands.shape
    # align
    d = quat_angle(q_cands, q_gt.unsqueeze(1))  # [B,K]
    align_loss = d.mean()

    return align_loss

class candidates_loss(nn.Module):
    def __init__(self,):
        super(candidates_loss, self).__init__()
        self.quan_norm = quat_normalize

    def forward(self, q_cands, q_gt):

        q_cands = self.quan_norm(q_cands)
        q_gt = self.quan_norm(q_gt)
        B, K, _ = q_cands.shape

        # align
        d = quat_angle(q_cands, q_gt.unsqueeze(1))  # [B,K]
        rot_loss = d.mean()

        total = rot_loss

        loss_dict = {
                        "rot_loss": rot_loss,
                        "total": total
                    }

        return total, loss_dict

class simplified_joint_loss(nn.Module):
    def __init__(self,):
        super().__init__()
        self.l2 = nn.MSELoss()
        self.t_loss = nn.L1Loss()

    def trans_loss(self, pred_translation, gt_translation):
        return self.t_loss(pred_translation, gt_translation)

    def forward(self, gt_quaternion, gt_translation,pred_quaternion,pred_translation):
        ang_q = quat_angle(pred_quaternion, gt_quaternion)
        rot_loss = ang_q.mean()
        trans_loss = self.trans_loss(pred_translation, gt_translation)

        loss_dict = {
            'rot_loss': rot_loss,
            'trans_loss': trans_loss,
        }

        return loss_dict
