"""
@Note: Modified from the original GitHub project: https://github.com/Lynne-Zheng-Linfang/HS-Pose
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# from losses.fs_net_loss import FLAGS


def get_neighbor_index(vertices: "(bs, vertice_num, 3)", neighbor_num: int):

    inner = torch.bmm(vertices, vertices.transpose(1, 2))
    quadratic = torch.sum(vertices ** 2, dim=2)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k=neighbor_num + 1, dim=-1, largest=False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index

def indexing_neighbor_new(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)"):
    bs, num_points, num_dims = tensor.size()
    idx_base = torch.arange(0, bs, device=tensor.device).view(-1, 1, 1) * num_points
    idx = index + idx_base
    idx = idx.view(-1)
    feature = tensor.reshape(bs * num_points, -1)[idx, :]
    _, out_num_points, n = index.size()
    feature = feature.view(bs, out_num_points, n, num_dims)
    return feature

def get_neighbor_direction_norm(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, neighbor_num)", return_unnormed = False):

    neighbors = indexing_neighbor_new(vertices, neighbor_index)
    neighbor_direction = neighbors - vertices.unsqueeze(2)
    neighbor_direction_norm = F.normalize(neighbor_direction, dim=-1)
    if return_unnormed:
        return neighbor_direction_norm.float(), neighbor_direction
    else:
        return neighbor_direction_norm.float()

def get_ORL_global(feature:'(bs, vertice_num, in_channel)', vertices: '(bs, vertice_num, 3)',
                          neighbor_num:'int'):
    vertice_num = feature.size(1)
    neighbor_index = get_neighbor_index(vertices, neighbor_num)
    feature = indexing_neighbor_new(feature, neighbor_index) # batch_size, num_points, k, num_dims
    feature = torch.max(feature, dim=2)[0]
    f_global = torch.mean(feature, dim=1, keepdim=True).repeat(1, vertice_num, 1)
    return f_global


class HyperS3_surface(nn.Module):

    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num


        self.STE_layer = nn.Conv1d(3, kernel_num, kernel_size=1, bias=False)


        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))


        self.conv2 = nn.Conv1d(2 * kernel_num, kernel_num, kernel_size=1, bias=False)


        self.align_conf_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.support_num * self.kernel_num)
        self.directions.data.uniform_(-std, std)
        for m in self.align_conf_mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

    def _compute_local_frame(self, S: torch.Tensor):

        bs, v, _, _ = S.size()
        device = S.device


        v0 = torch.randn(bs, v, 3, 1, device=device)
        e1 = torch.matmul(S, v0).squeeze(-1)
        e1 = F.normalize(e1, dim=-1)

        tmp = torch.tensor([1., 0., 0.], device=device).view(1, 1, 3).expand(bs, v, 3)
        e2 = torch.cross(e1, tmp, dim=-1)
        mask = e2.norm(dim=-1) < 1e-6
        tmp2 = torch.tensor([0., 1., 0.], device=device).view(1, 1, 3).expand(bs, v, 3)
        e2[mask] = torch.cross(e1[mask], tmp2[mask], dim=-1)
        e2 = F.normalize(e2, dim=-1)

        e3 = torch.cross(e1, e2, dim=-1)
        e3 = F.normalize(e3, dim=-1)

        E = torch.stack([e1, e2, e3], dim=-1)
        return E

    def _covariance_and_confidence(self, R: torch.Tensor):

        bs, v, k, _ = R.size()
        device = R.device


        outer = R.unsqueeze(-1) * R.unsqueeze(-2)
        S = outer.mean(dim=2)


        trace = S[..., 0, 0] + S[..., 1, 1] + S[..., 2, 2]


        eye = torch.eye(3, device=device).view(1, 1, 3, 3)
        iso = (trace / 3.0).view(bs, v, 1, 1) * eye
        residual = S - iso
        anisotropy = residual.pow(2).sum(dim=(-1, -2))

        geom = torch.stack([trace, anisotropy], dim=-1)
        geom_flat = geom.view(bs * v, 2)
        alpha = torch.sigmoid(self.align_conf_mlp(geom_flat)).view(bs, v, 1)

        return S, alpha

    def forward(self, vertices, neighbor_num):

        bs, v, _ = vertices.size()


        f_STE = self.STE_layer(vertices.transpose(1, 2)).transpose(1, 2)


        idx = get_neighbor_index(vertices, neighbor_num)
        neigh = indexing_neighbor_new(vertices, idx)
        R = neigh - vertices.unsqueeze(2)


        S, alpha = self._covariance_and_confidence(R)


        E = self._compute_local_frame(S)

        R_local = torch.matmul(
            E.transpose(-1, -2).unsqueeze(2),
            R.unsqueeze(-1)
        ).squeeze(-1)


        receptive_aligned = F.normalize(R_local, dim=-1)
        receptive_euclid  = F.normalize(R,       dim=-1)


        support_direction_norm = F.normalize(self.directions, dim=0)


        theta_align = torch.matmul(receptive_aligned, support_direction_norm)
        theta_align = F.relu(theta_align)
        theta_align = theta_align.view(bs, v, neighbor_num, self.support_num, self.kernel_num)

        theta_align = theta_align.max(dim=2)[0]

        feature_align = theta_align.mean(dim=2)


        theta_eucl = torch.matmul(receptive_euclid, support_direction_norm)
        theta_eucl = F.relu(theta_eucl)
        theta_eucl = theta_eucl.view(bs, v, neighbor_num, self.support_num, self.kernel_num)

        theta_eucl = theta_eucl.max(dim=2)[0]

        feature_eucl = theta_eucl.mean(dim=2)

        feature = alpha * feature_align + (1.0 - alpha) * feature_eucl

        f_global = get_ORL_global(feature, vertices, neighbor_num)
        feat_cat = torch.cat([feature, f_global], dim=-1)
        feat_cat = self.conv2(feat_cat.transpose(1, 2)).transpose(1, 2)

        return feat_cat + f_STE


class HyperS3(nn.Module):


    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num


        self.STE_layer = nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False)


        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))


        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))

        self.conv2 = nn.Conv1d(2 * out_channel, out_channel, kernel_size=1, bias=False)

        self.align_conf_mlp = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1)
        )

        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)
        self.directions.data.uniform_(-std, std)

        for m in self.align_conf_mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                fan, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / math.sqrt(fan)
                nn.init.uniform_(m.bias, -bound, bound)

    def _compute_local_frame(self, S):
        bs, v, _, _ = S.size()
        device = S.device

        v0 = torch.randn(bs, v, 3, 1, device=device)
        e1 = torch.matmul(S, v0).squeeze(-1)
        e1 = F.normalize(e1, dim=-1)

        tmp = torch.tensor([1., 0., 0.], device=device).view(1,1,3).expand(bs, v, 3)
        e2 = torch.cross(e1, tmp, dim=-1)
        mask = e2.norm(dim=-1) < 1e-6
        tmp2 = torch.tensor([0.,1.,0.], device=device).view(1,1,3).expand(bs, v, 3)
        e2[mask] = torch.cross(e1[mask], tmp2[mask], dim=-1)
        e2 = F.normalize(e2, dim=-1)

        e3 = F.normalize(torch.cross(e1, e2, dim=-1), dim=-1)

        return torch.stack([e1, e2, e3], dim=-1)

    def _covariance_and_confidence(self, R):
        bs, v, k, _ = R.size()
        device = R.device

        outer = R.unsqueeze(-1) * R.unsqueeze(-2)
        S = outer.mean(dim=2)

        trace = S[...,0,0] + S[...,1,1] + S[...,2,2]

        eye = torch.eye(3, device=device).view(1,1,3,3)
        iso = (trace/3.0).view(bs,v,1,1) * eye
        anisotropy = (S - iso).pow(2).sum(dim=(-1,-2))

        geom = torch.stack([trace, anisotropy], dim=-1)
        alpha = torch.sigmoid(self.align_conf_mlp(geom.view(bs*v,2))).view(bs, v, 1)

        return S, alpha

    def forward(self, vertices, feature_map, neighbor_num):

        bs, v, _ = vertices.size()


        f_STE = self.STE_layer(feature_map.transpose(1, 2)).transpose(1, 2)


        neighbor_index = get_neighbor_index(feature_map, neighbor_num)
        neighbors = indexing_neighbor_new(vertices, neighbor_index)
        R = neighbors - vertices.unsqueeze(2)

        S, alpha = self._covariance_and_confidence(R)

        E = self._compute_local_frame(S)
        R_local = torch.matmul(E.transpose(-1,-2).unsqueeze(2),
                               R.unsqueeze(-1)).squeeze(-1)

        receptive_aligned = F.normalize(R_local, dim=-1)
        receptive_euclid  = F.normalize(R,       dim=-1)

        support_direction_norm = F.normalize(self.directions, dim=0)

        theta_align = torch.matmul(receptive_aligned, support_direction_norm)
        theta_align = F.relu(theta_align)
        theta_align = theta_align.view(bs,v,neighbor_num,self.support_num,self.out_channel)
        theta_align = torch.max(theta_align, dim=2)[0].mean(dim=2)

        theta_eucl = torch.matmul(receptive_euclid, support_direction_norm)
        theta_eucl = F.relu(theta_eucl)
        theta_eucl = theta_eucl.view(bs,v,neighbor_num,self.support_num,self.out_channel)
        theta_eucl = torch.max(theta_eucl, dim=2)[0].mean(dim=2)


        theta = alpha * theta_align + (1 - alpha) * theta_eucl

        feature_w = feature_map @ self.weights + self.bias
        feature_center = feature_w[:, :, :self.out_channel]
        feature_support = feature_w[:, :, self.out_channel:]

        feature_support = indexing_neighbor_new(feature_support, neighbor_index)
        feature_support = feature_support.view(bs, v, neighbor_num, self.support_num, self.out_channel)

        activation_support = theta.unsqueeze(2).unsqueeze(2) * feature_support
        activation_support = activation_support.max(dim=2)[0].mean(dim=2)

        feature = feature_center + activation_support


        f_global = get_ORL_global(feature, vertices, neighbor_num)
        feat_cat = torch.cat([feature, f_global], dim=-1)
        feat_cat = self.conv2(feat_cat.transpose(1,2)).transpose(1,2)

        return feat_cat + f_STE




class Pool_layer_SO3(nn.Module):

    def __init__(self, pooling_rate: int = 4, neighbor_num: int = 4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        bs, vertice_num, _ = vertices.size()

        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        neighbor_feature = indexing_neighbor_new(feature_map, neighbor_index)
        pooled_feature = torch.max(neighbor_feature, dim=2)[0]

        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :]
        feature_map_pool = pooled_feature[:, sample_idx, :]

        return vertices_pool, feature_map_pool

