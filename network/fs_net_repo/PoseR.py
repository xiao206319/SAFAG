import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import absl.flags as flags
from absl import app

EPS = 1e-8

FLAGS = flags.FLAGS

def safe_normalize(x, dim=-1, eps=1e-6):
    return x / (torch.norm(x, dim=dim, keepdim=True) + eps)

class ConditionalQuaternionSampler(nn.Module):

    def __init__(self, feat_dim=1290, num_candidates=64, noise_dim=8, hidden_dim=256):
        super().__init__()
        self.K = num_candidates
        self.noise_dim = noise_dim

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.mlp = nn.Sequential(
            nn.Linear(feat_dim + noise_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4)
        )

    def forward(self, global_feat):

        B = global_feat.size(0)


        pooled = self.global_pool(global_feat)
        pooled = pooled.squeeze(-1)

        noise = torch.randn(B, self.K, self.noise_dim, device=global_feat.device)

        pooled_expand = pooled.unsqueeze(1).expand(-1, self.K, -1)
        cond_input = torch.cat([pooled_expand, noise], dim=-1)

        cond_input = cond_input.view(B * self.K, -1)
        quat_raw = self.mlp(cond_input)

        if torch.isnan(quat_raw).any() or torch.isinf(quat_raw).any():
            q_min = torch.nan_to_num(quat_raw).min().item()
            q_max = torch.nan_to_num(quat_raw).max().item()

        quat_norm = safe_normalize(quat_raw, dim=-1)

        if torch.isnan(quat_norm).any() or torch.isinf(quat_norm).any():
            n_min = torch.nan_to_num(quat_norm).min().item()
            n_max = torch.nan_to_num(quat_norm).max().item()

        candidates = quat_norm.view(B, self.K, 4)

        return candidates


def grad_nan_hook(grad, name):
    if torch.isnan(grad).any() or torch.isinf(grad).any():
        print(f"[BACKWARD NaN] {name} grad has NaN/Inf!")
        import ipdb; ipdb.set_trace()
    return grad


class Quaternion_candidates_generator(nn.Module):
    def __init__(self):
        super(Quaternion_candidates_generator, self).__init__()
        self.f = FLAGS.feat_c_R
        self.k = 64
        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = x.squeeze(2)
        x = x.view(-1, self.k, 4)

        x = x.contiguous()

        return x



class final_quaternion(nn.Module):
    def __init__(self):
        super(final_quaternion, self).__init__()
        self.f = FLAGS.feat_c_R
        self.k = FLAGS.R_c

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)
        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 4, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()

        return x


def quat_normalize(q):
    return q / (q.norm(dim=-1, keepdim=True) + EPS)

def quat_conjugate(q):
    w,x,y,z = q.unbind(-1)
    return torch.stack([w, -x, -y, -z], dim=-1)

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1.unbind(-1)
    w2,x2,y2,z2 = q2.unbind(-1)
    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def quat_mean(qcands):

    B,K,_ = qcands.shape
    m = qcands.mean(dim=1, keepdim=True)
    dots = (qcands * m).sum(-1, keepdim=True)
    signs = torch.where(dots >= 0, 1.0, -1.0)
    aligned = qcands * signs
    mean = aligned.mean(dim=1)
    return quat_normalize(mean)

def quat_log_map(q):

    w = q[...,0:1]
    v = q[...,1:]
    vnorm = v.norm(dim=-1, keepdim=True)

    theta = 2.0 * torch.atan2(vnorm, w.clamp(min=EPS))

    u = v / (vnorm + EPS)
    return (u * theta)


def compute_second_moment(qcands):
    B,K,_ = qcands.shape
    q_col = qcands.unsqueeze(-1)
    qT = qcands.unsqueeze(-2)

    M = torch.matmul(q_col, qT).mean(dim=1)
    return M

def spectral_features_of_M(M, topk=3):

    eigvals, eigvecs = torch.linalg.eigh(M)
    eigvals = eigvals.flip(-1)
    eigvecs = eigvecs.flip(-1)

    return eigvals[:,:topk], eigvecs[:,:,:topk]

class GlobalConvEncoder(nn.Module):
    def __init__(self, g_in, hidden, Dg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(g_in, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden, Dg, kernel_size=1),
            nn.ReLU()
        )

        for m in self.net.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(-1)
        out = self.net(x)
        return out.squeeze(-1)


class CandidateDistributionEncoder(nn.Module):

    def __init__(self, Dg=256, Dk=64, hidden=128):
        super().__init__()
        self.Dg = Dg; self.Dk = Dk

        self.pc_mlp = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, Dk), nn.ReLU()
        )

        g_in = 3 + 1 + 3 + 12
        self.global_mlp = GlobalConvEncoder(g_in, hidden, Dg)

        self.pc_project = nn.Linear(Dk, Dk)

    def forward(self, qcands):

        B,K,_ = qcands.shape
        qc = quat_normalize(qcands)


        q_mean = quat_mean(qc)
        q_mean_inv = quat_conjugate(q_mean)

        q_mean_inv_exp = q_mean_inv.unsqueeze(1).expand(-1,K,-1)
        r = quat_mul(q_mean_inv_exp, qc)


        r_log = quat_log_map(r)

        r_log_flat = r_log.view(B*K, 3)
        per_emb = self.pc_mlp(r_log_flat).view(B, K, -1)
        per_emb = self.pc_project(per_emb)


        mean_log = r_log.mean(dim=1)
        mean_dot2 = ( (qc * q_mean.unsqueeze(1)).sum(dim=-1).abs() ** 2 ).mean(dim=1, keepdim=True)
        M = compute_second_moment(r)
        eigvals, eigvecs = spectral_features_of_M(M, topk=3)
        eigvecs_flat = eigvecs.reshape(B, -1)

        global_input = torch.cat([mean_log, mean_dot2, eigvals, eigvecs_flat], dim=-1)
        global_input = F.normalize(global_input, dim=-1)


        for name, p in self.global_mlp.named_parameters():
            if not torch.isfinite(p).all():
                print("PARAM BAD:", name, "has NaN/Inf")
        global_emb = self.global_mlp(global_input)
        if torch.isnan(global_emb).any() or torch.isinf(global_emb).any():
            print(global_emb)

        return {
            "global_emb": global_emb,
            "per_emb": per_emb,
            "q_mean": q_mean,
            "M": M
        }


class FeatureFusion(nn.Module):

    def __init__(self, in_C=1290, Dg=256, Dk=64, mode='film', out_C=None):
        super().__init__()
        self.mode = mode
        self.in_C = in_C
        self.out_C = out_C if out_C is not None else in_C
        if mode == 'film':

            self.film_scale = nn.Linear(Dg, self.in_C)
            self.film_bias  = nn.Linear(Dg, self.in_C)

            self.post = nn.Sequential(nn.Linear(self.in_C, self.out_C), nn.ReLU())
        elif mode == 'attn':

            self.num_heads = 8
            self.q_proj = nn.Linear(self.in_C, self.in_C)
            self.k_proj = nn.Linear(Dk + Dg, self.in_C)
            self.v_proj = nn.Linear(Dk + Dg, self.in_C)
            self.out = nn.Linear(self.in_C, self.out_C)
        else:
            raise ValueError("mode must be 'film' or 'attn'")

    def forward(self, point_feats, encoder_out):

        B,N,C = point_feats.shape
        global_emb = encoder_out["global_emb"]
        per_emb = encoder_out["per_emb"]

        if self.mode == 'film':
            scale = self.film_scale(global_emb).unsqueeze(1)
            bias  = self.film_bias(global_emb).unsqueeze(1)
            fused = point_feats * (1 + scale) + bias
            out = self.post(fused)
            return out

        else:

            B,K,Dk = per_emb.shape
            glob_rep = global_emb.unsqueeze(1).expand(-1, K, -1)
            kv = torch.cat([per_emb, glob_rep], dim=-1)

            Q = self.q_proj(point_feats)
            Kp = self.k_proj(kv)
            Vp = self.v_proj(kv)

            attn = torch.matmul(Q, Kp.transpose(-1,-2)) / math.sqrt(C)
            attn = F.softmax(attn, dim=-1)
            agg = torch.matmul(attn, Vp)
            out = self.out(agg)
            return out


def axis_angle_to_quat(axis_angle):
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    small_angle = angle < 1e-8
    axis = axis_angle / (angle + 1e-8)

    half_angle = 0.5 * angle
    qw = torch.cos(half_angle)
    q_xyz = axis * torch.sin(half_angle)

    q = torch.cat([qw, q_xyz], dim=-1)
    q = torch.where(small_angle.expand_as(q),
                    torch.tensor([1.0, 0.0, 0.0, 0.0], device=q.device, dtype=q.dtype),
                    q)
    return q

class AnchorLevelPredictor(nn.Module):
    def __init__(self, in_channels=1290, hidden=512, K=64):

        super().__init__()
        self.K = K
        self.fc1 = nn.Linear(in_channels, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.delta_head = nn.Linear(hidden, K * 3)
        self.out_head = nn.Linear(K*4, 4)

    def forward(self, pts_feat, quaternion_candidates):

        B = pts_feat.shape[0]

        global_feat = torch.max(pts_feat, dim=2)[0]
        x = F.relu(self.fc1(global_feat))
        x = F.relu(self.fc2(x))

        delta = self.delta_head(x)
        delta = delta.view(B, self.K, 3)

        q_delta = axis_angle_to_quat(delta)

        q_corrected = quat_mul(quaternion_candidates, q_delta)
        q_corrected = F.normalize(q_corrected, dim=-1)
        q_corrected = q_corrected.view(B, self.K*4)
        final_q = self.out_head(q_corrected)
        final_q = F.normalize(final_q, dim=-1)

        return final_q

def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module

class QuaternionRefineHead(nn.Module):

    def __init__(self, feat_dim=1290, quat_dim=4, pose_emb_dim=0):
        super().__init__()
        self.use_pose_emb = pose_emb_dim > 0
        input_dim = feat_dim + quat_dim + pose_emb_dim

        self.act = nn.ReLU(inplace=True)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            self.act,
            nn.Linear(256, 256),
            self.act,
            zero_module(nn.Linear(256, quat_dim))
        )

    def forward(self, feat, coarse_q, pose_emb=None):
        if self.use_pose_emb:
            assert pose_emb is not None
            x = torch.cat([feat, coarse_q, pose_emb], dim=-1)
        else:
            x = torch.cat([feat, coarse_q], dim=-1)

        delta_q = self.net(x)
        return delta_q

class PoseEmbedding(nn.Module):
    def __init__(self, in_dim=4, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embed_dim)
        )

    def forward(self, q):
        return self.net(q)

def quat_to_axis_angle(q):


    q = F.normalize(q, dim=-1)
    w, v = q[:, 0], q[:, 1:]
    angle = 2 * torch.acos(w.clamp(-1+1e-7, 1-1e-7))
    axis = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    return axis, angle.unsqueeze(-1)