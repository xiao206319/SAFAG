import torch
import torch.nn as nn
import torch.nn.functional as F
from absl import flags
import itertools

FLAGS = flags.FLAGS


def quat_angle(q1, q2):
    q1 = F.normalize(q1, dim=-1)
    q2 = F.normalize(q2, dim=-1)
    dots = torch.sum(q1 * q2, dim=-1).clamp(-1+ 1e-6, 1 -1e-6)
    return 2 * torch.acos(dots.abs())


def axis_angle_to_quat(axis, angle):

    if not torch.is_tensor(angle):
        angle = torch.tensor(angle, device=axis.device, dtype=axis.dtype)
    else:
        angle = angle.to(device=axis.device, dtype=axis.dtype)


    if axis.ndim == 1:
        axis = axis / (axis.norm() + 1e-8)
        half = angle / 2
        c = torch.cos(half)
        s = torch.sin(half)
        q = torch.cat([c.unsqueeze(0), s * axis], dim=0)
        return q


    elif axis.ndim == 2:
        B = axis.shape[0]
        axis = axis / (axis.norm(dim=-1, keepdim=True) + 1e-8)
        if angle.ndim == 0:
            angle = angle.expand(B)
        half = angle / 2
        c = torch.cos(half).unsqueeze(-1)
        s = torch.sin(half).unsqueeze(-1)
        q = torch.cat([c, s * axis], dim=-1)
        return q

    else:
        raise ValueError(f"Invalid axis shape: {axis.shape}")



def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

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

class SymAxisGMM(nn.Module):
    def __init__(self, n_components=3, hidden_rot=64, hidden_global=512, hidden_fuse=256):
        super().__init__()
        self.n_components = n_components


        self.fc_rot = nn.Sequential(
            nn.Linear(4, hidden_rot),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_rot, hidden_rot),
            nn.ReLU(inplace=True)
        )

        self.fc_global = nn.Sequential(
            nn.Linear(1290, hidden_global),
            nn.ReLU(inplace=True)
        )

        self.fc_fuse = nn.Sequential(
            nn.Linear(hidden_rot + hidden_global, hidden_fuse),
            nn.ReLU(inplace=True)
        )

        self.fc_pi = nn.Linear(hidden_fuse, n_components)

        self.register_buffer("axes", torch.eye(3))

    def forward(self, rot_feat, feat):

        rot_emb = self.fc_rot(rot_feat)
        f_global = feat.max(dim=1)[0]
        f_global = self.fc_global(f_global)

        fuse = torch.cat([rot_emb, f_global], dim=-1)
        h = self.fc_fuse(fuse)


        pi_logits = self.fc_pi(h)
        log_probs = F.log_softmax(pi_logits , dim=-1)
        pi_probs = log_probs.exp()

        weighted_axis = torch.matmul(pi_probs, self.axes.to(rot_feat.device))
        weighted_axis = F.normalize(weighted_axis, dim=-1)

        return weighted_axis, pi_probs, self.axes, log_probs


class SymmetryAwareLoss(nn.Module):
    def __init__(self, num_samples=36, eps=1e-8):
        """
        num_samples: 每个轴采样角度数
        eps: 数值稳定项
        """
        super().__init__()
        self.num_samples = num_samples
        self.eps = eps
        self.register_buffer("angles", torch.linspace(0, 2 * torch.pi, steps=num_samples))

    def quat_l2_min_sign(self, q1, q2):
        """
        q1, q2: [B,4]
        返回对应的 min-sign L2 loss
        """
        diff1 = (q1 - q2) ** 2
        diff2 = (q1 + q2) ** 2
        return torch.min(diff1.sum(dim=-1), diff2.sum(dim=-1))

    def forward(self, pred_q, gt_q, pi_probs, axes, weighted_axis, log_probs=None):
        """
        pred_q:        [B,4]
        gt_q:          [B,4]
        pi_probs:      [B,3]
        axes:          [3,3]
        weighted_axis: [B,3]
        """
        weighted_axis = F.normalize(weighted_axis, dim=-1, eps=self.eps)

        errs_per_angle = []

        for ang in self.angles:
            # 正向旋转
            q_rot_pos = axis_angle_to_quat(weighted_axis, ang)  # [B,4]
            # 反向旋转
            q_rot_neg = axis_angle_to_quat(-weighted_axis, ang)  # [B,4]

            # 生成两组等价解
            q_equiv_pos = quat_mul(gt_q, q_rot_pos)
            q_equiv_neg = quat_mul(gt_q, q_rot_neg)

            # 分别计算误差
            err_pos = quat_angle(pred_q, q_equiv_pos)  # [B]
            err_neg = quat_angle(pred_q, q_equiv_neg)  # [B]

            # 两方向取最小
            err_min = torch.minimum(err_pos, err_neg)
            errs_per_angle.append(err_min)

        errs_per_angle = torch.stack(errs_per_angle, dim=-1)
        min_err, _ = torch.min(errs_per_angle, dim=-1)  # [B]
        loss_gmm = min_err.mean()

        if log_probs is None:
            log_probs = torch.log(pi_probs + 1e-8)

        return loss_gmm

class AdaptiveSymmetryPredictor(nn.Module):
    def __init__(self, n_components=3, hidden_rot=64, hidden_global=512, hidden_fuse=256, alpha=10.0):
        super().__init__()
        self.n_components = n_components
        self.alpha = alpha

        self.fc_rot = nn.Sequential(
            nn.Linear(4, hidden_rot),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_rot, hidden_rot),
            nn.ReLU(inplace=True)
        )


        self.fc_global = nn.Sequential(
            nn.Linear(1290, hidden_global),
            nn.ReLU(inplace=True)
        )

        self.fc_fuse = nn.Sequential(
            nn.Linear(hidden_rot + hidden_global, hidden_fuse),
            nn.ReLU(inplace=True)
        )

        self.fc_pi = nn.Linear(hidden_fuse, n_components)

        self.register_buffer("axes", torch.eye(3))

        self.fc_n2 = nn.Sequential(
            nn.Linear(hidden_fuse + 3, hidden_fuse),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_fuse, 3)
        )


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


    def forward(self, rot_feat, feat, pts=None,gt_R=None):

        B = rot_feat.shape[0]
        device = rot_feat.device


        rot_emb = self.fc_rot(rot_feat)
        f_global = feat.max(dim=1)[0]
        f_global = self.fc_global(f_global)
        fuse = torch.cat([rot_emb, f_global], dim=-1)
        h = self.fc_fuse(fuse)

        pi_logits = self.fc_pi(h)
        pi_probs = F.softmax(pi_logits, dim=-1)

        axes = self.axes.to(device)
        n1 = torch.matmul(pi_probs, axes)
        n1 = F.normalize(n1, dim=-1)

        n2_raw = self.fc_n2(torch.cat([h, n1], dim=-1))
        v2_proj = n2_raw - torch.sum(n2_raw * n1, dim=-1, keepdim=True) * n1
        n2 = F.normalize(v2_proj, dim=-1)

        n3 = F.normalize(torch.cross(n1, n2, dim=-1), dim=-1)

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

        n1 = F.normalize(n1, dim=-1)
        n2 = F.normalize(n2, dim=-1)
        n3 = F.normalize(n3, dim=-1)


        if pts is not None:
            loss_x = self.mirror_consistency_loss(pts, n1_obj)
            loss_y = self.mirror_consistency_loss(pts, n2_obj)
            loss_z = self.mirror_consistency_loss(pts, n3_obj)
            geom_losses = torch.stack([loss_x, loss_y, loss_z], dim=-1)


            inv_scores = 1.0 / (geom_losses + 1e-6)
            geom_weights = inv_scores / (inv_scores.sum(dim=-1, keepdim=True) + 1e-8)

        sym_normals=(n1, n2, n3)

        return sym_normals, geom_weights, pi_probs


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
                S = S @ S_list[k]
        S_combos.append(S)

    R_equivs = [R_gt @ S for S in S_combos]
    R_equivs = torch.stack(R_equivs, dim=1)
    return R_equivs


class SymMirrorAwareLoss(nn.Module):
    def __init__(self, eps=1e-6, mirror_weight=2.0):
        super().__init__()
        self.eps = eps
        self.mirror_weight = mirror_weight



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

    def forward(self, pred_q, gt_q,gt_R, pts, sym_normals, geom_weights=None):

        B = pred_q.shape[0]
        device = pred_q.device
        pred_q = F.normalize(pred_q, dim=-1)

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
            geom_loss = loss_n1.mean()
        elif sym_plane_count == 2:
            geom_loss = ((loss_n1 + loss_n2) / 2).mean()
        else:
            geom_loss = ((loss_n1 + loss_n2 + loss_n3) / 3).mean()


        if sym_plane_count == 1:
            normals_used = [n1]
        elif sym_plane_count == 2:
            normals_used = [n1, n2]
        else:
            normals_used = [n1, n2, n3]

        R_equivs = generate_equiv_poses(gt_R, normals_used)


        R_pred = quaternion_to_rotation_matrix(pred_q)


        B, N_eq = R_equivs.shape[:2]
        errs_all = []
        for i in range(N_eq):
            R_eq_i = R_equivs[:, i, :, :]
            M = torch.bmm(R_pred.transpose(1, 2), R_eq_i)
            trace = M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2]
            cos_theta = ((trace - 1.0) / 2.0).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            angle_deg = torch.acos(cos_theta)
            errs_all.append(angle_deg)

        errs_all = torch.stack(errs_all, dim=1)
        min_err = errs_all.min(dim=-1)[0]
        total_quat_loss = min_err.mean()


        total_loss =2 * total_quat_loss +   geom_loss

        return total_loss
