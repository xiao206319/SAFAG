import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
import absl.flags as flags
FLAGS = flags.FLAGS
def get_vertical_rot_vec(c1, c2, y, z):

    y = y.view(-1)
    z = z.view(-1)
    rot_x = torch.cross(y, z)
    rot_x = rot_x / (torch.norm(rot_x) + 1e-8)

    y_z_cos = torch.sum(y * z)
    y_z_theta = torch.acos(y_z_cos)
    theta_2 = c1 / (c1 + c2) * (y_z_theta - math.pi / 2)
    theta_1 = c2 / (c1 + c2) * (y_z_theta - math.pi / 2)

    c = torch.cos(theta_1)
    s = torch.sin(theta_1)
    rotmat_y = torch.tensor([[rot_x[0]*rot_x[0]*(1-c)+c, rot_x[0]*rot_x[1]*(1-c)-rot_x[2]*s, rot_x[0]*rot_x[2]*(1-c)+rot_x[1]*s],
                             [rot_x[1]*rot_x[0]*(1-c)+rot_x[2]*s, rot_x[1]*rot_x[1]*(1-c)+c, rot_x[1]*rot_x[2]*(1-c)-rot_x[0]*s],
                             [rot_x[0]*rot_x[2]*(1-c)-rot_x[1]*s, rot_x[2]*rot_x[1]*(1-c)+rot_x[0]*s, rot_x[2]*rot_x[2]*(1-c)+c]]).to(y.device)
    new_y = torch.mm(rotmat_y, y.view(-1, 1))

    c = torch.cos(-theta_2)
    s = torch.sin(-theta_2)
    rotmat_z = torch.tensor([[rot_x[0] * rot_x[0] * (1 - c) + c, rot_x[0] * rot_x[1] * (1 - c) - rot_x[2] * s,
                              rot_x[0] * rot_x[2] * (1 - c) + rot_x[1] * s],
                             [rot_x[1] * rot_x[0] * (1 - c) + rot_x[2] * s, rot_x[1] * rot_x[1] * (1 - c) + c,
                              rot_x[1] * rot_x[2] * (1 - c) - rot_x[0] * s],
                             [rot_x[0] * rot_x[2] * (1 - c) - rot_x[1] * s,
                              rot_x[2] * rot_x[1] * (1 - c) + rot_x[0] * s, rot_x[2] * rot_x[2] * (1 - c) + c]]).to(
        z.device)

    new_z = torch.mm(rotmat_z, z.view(-1, 1))
    return new_y.view(-1), new_z.view(-1)

def get_rot_mat_y_first(y, x):


    y = F.normalize(y, p=2, dim=-1)
    z = torch.cross(x, y, dim=-1)
    z = F.normalize(z, p=2, dim=-1)
    x = torch.cross(y, z, dim=-1)


    return torch.stack((x, y, z), dim=-1)


def batch_orthogonalize_and_get_rotation_matrix(green_vecs, red_vecs):

    y = F.normalize(green_vecs, dim=-1)

    z = F.normalize(red_vecs, dim=-1)

    x = torch.cross(y, z, dim=-1)
    x = F.normalize(x, dim=-1)

    z = torch.cross(x, y, dim=-1)
    z = F.normalize(z, dim=-1)

    rot_mat = torch.stack((x, y, z), dim=-1)
    return rot_mat

def get_rot_vec_vert_batch(c1, c2, y, z):
    bs = c1.shape[0]
    new_y = y
    new_z = z
    for i in range(bs):
        new_y[i, ...], new_z[i, ...] = get_vertical_rot_vec(c1[i, ...], c2[i, ...], y[i, ...], z[i, ...])
    return new_y, new_z

def rotation_matrix_to_quaternion(rot_matrices, eps=1e-8):

    R = rot_matrices
    if R.dim() != 3 or R.size(-2) != 3 or R.size(-1) != 3:
        raise ValueError("rot_matrices must be shape [B,3,3]")

    B = R.shape[0]
    R00 = R[:, 0, 0]
    R11 = R[:, 1, 1]
    R22 = R[:, 2, 2]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]

    trace = R00 + R11 + R22

    s1_arg = trace + 1.0
    s1 = torch.sqrt(torch.clamp(s1_arg, min=0.0)) * 2.0
    qw1 = 0.25 * s1
    qx1 = (R21 - R12) / (s1 + eps)
    qy1 = (R02 - R20) / (s1 + eps)
    qz1 = (R10 - R01) / (s1 + eps)

    s2_arg = 1.0 + R00 - R11 - R22
    s2 = torch.sqrt(torch.clamp(s2_arg, min=0.0)) * 2.0
    qw2 = (R21 - R12) / (s2 + eps)
    qx2 = 0.25 * s2
    qy2 = (R01 + R10) / (s2 + eps)
    qz2 = (R02 + R20) / (s2 + eps)

    s3_arg = 1.0 + R11 - R00 - R22
    s3 = torch.sqrt(torch.clamp(s3_arg, min=0.0)) * 2.0
    qw3 = (R02 - R20) / (s3 + eps)
    qx3 = (R01 + R10) / (s3 + eps)
    qy3 = 0.25 * s3
    qz3 = (R12 + R21) / (s3 + eps)

    s4_arg = 1.0 + R22 - R00 - R11
    s4 = torch.sqrt(torch.clamp(s4_arg, min=0.0)) * 2.0
    qw4 = (R10 - R01) / (s4 + eps)
    qx4 = (R02 + R20) / (s4 + eps)
    qy4 = (R12 + R21) / (s4 + eps)
    qz4 = 0.25 * s4

    cond1 = trace > 0.0

    cond2 = (~cond1) & (R00 > R11) & (R00 > R22)

    cond3 = (~cond1) & (~cond2) & (R11 > R22)

    cond4 = ~(cond1 | cond2 | cond3)


    qw = torch.where(cond1, qw1,
                     torch.where(cond2, qw2,
                                 torch.where(cond3, qw3, qw4)))
    qx = torch.where(cond1, qx1,
                     torch.where(cond2, qx2,
                                 torch.where(cond3, qx3, qx4)))
    qy = torch.where(cond1, qy1,
                     torch.where(cond2, qy2,
                                 torch.where(cond3, qy3, qy4)))
    qz = torch.where(cond1, qz1,
                     torch.where(cond2, qz2,
                                 torch.where(cond3, qz3, qz4)))

    quats = torch.stack((qw, qx, qy, qz), dim=-1)

    quats = F.normalize(quats, p=2, dim=-1, eps=eps)
    return quats

if __name__ == '__main__':
    g_R=torch.tensor([[0.3126, 0.0018, -0.9499],
            [0.7303, -0.6400, 0.2391],
            [-0.6074, -0.7684, -0.2014]], device='cuda:0')
    y = g_R[:, 1]
    x = g_R[:, 0]
    c1 = 5
    c2 = 1
    y = y / torch.norm(y)
    x = x / torch.norm(x)
    L = torch.dot(y, x)
    Lp = torch.cross(x, y)
    Lp = Lp / torch.norm(Lp)
    new_y, nnew_x = get_vertical_rot_vec(c1, c2, y, x)
    M = torch.dot(new_y, nnew_x)
    Mp = torch.cross(new_y, nnew_x)
    Mp = Mp / torch.norm(Mp)
    new_R = get_rot_mat_y_first(new_y.view(1, -1), nnew_x.view(1, -1))
    print('OK')