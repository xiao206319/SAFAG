# Modified from FS-Net
import torch.nn as nn
import network.fs_net_repo.gcn3d as gcn3d
import torch
import torch.nn.functional as F
from absl import app
import absl.flags as flags
import network.fs_net_repo.s3_layer as s3_layer

FLAGS = flags.FLAGS


class FaceRecon(nn.Module):
    def __init__(self):
        super(FaceRecon, self).__init__()
        self.neighbor_num = FLAGS.gcn_n_num
        self.support_num = FLAGS.gcn_sup_num

        self.conv_0 = s3_layer.HyperS3_surface(kernel_num=128, support_num=self.support_num)
        self.conv_1 = s3_layer.HyperS3(128, 128, support_num=self.support_num)
        self.pool_1 = s3_layer.Pool_layer_SO3(pooling_rate=4, neighbor_num=4)
        self.conv_2 = s3_layer.HyperS3(128, 256, support_num=self.support_num)
        self.conv_3 = s3_layer.HyperS3(256, 256, support_num=self.support_num)
        self.pool_2 = s3_layer.Pool_layer_SO3(pooling_rate=4, neighbor_num=4)
        self.conv_4 = s3_layer.HyperS3(256, 512, support_num=self.support_num)

        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)

        self.recon_num = 3
        self.face_recon_num = FLAGS.face_recon_c

        dim_fuse = sum([128, 128, 256, 256, 512, FLAGS.obj_c])


        if FLAGS.train:
            self.conv1d_block = nn.Sequential(
                nn.Conv1d(dim_fuse, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
            )

            self.recon_head = nn.Sequential(
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, self.recon_num, 1),
            )

            self.face_head = nn.Sequential(
                nn.Conv1d(FLAGS.feat_face + 3, 512, 1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Conv1d(512, 256, 1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 128, 1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, self.face_recon_num, 1),
            )

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                cat_id: "tensor (bs, 1)",
                ):

        bs, vertice_num, _ = vertices.size()

        if cat_id.shape[0] == 1:
            obj_idh = cat_id.view(-1, 1).repeat(cat_id.shape[0], 1)
        else:
            obj_idh = cat_id.view(-1, 1)

        one_hot = torch.zeros(bs, FLAGS.obj_c).to(cat_id.device).scatter_(1, obj_idh.long(), 1)

        fm_0 = F.relu(self.conv_0(vertices, self.neighbor_num), inplace=True)
        fm_1 = F.relu(self.bn1(self.conv_1(vertices, fm_0, self.neighbor_num).transpose(1, 2)).transpose(1, 2), inplace=True)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        fm_2 = F.relu(self.bn2(self.conv_2(v_pool_1, fm_pool_1,
                                           min(self.neighbor_num, v_pool_1.shape[1] // 8)).transpose(1, 2)).transpose(1, 2), inplace=True)
        fm_3 = F.relu(self.bn3(self.conv_3(v_pool_1, fm_2,
                                           min(self.neighbor_num, v_pool_1.shape[1] // 8)).transpose(1, 2)).transpose(1, 2), inplace=True)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        fm_4 = self.conv_4(v_pool_2, fm_pool_2, min(self.neighbor_num, v_pool_2.shape[1] // 8))
        f_global = fm_4.max(1)[0]

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor_new(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor_new(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor_new(fm_4, nearest_pool_2).squeeze(2)
        one_hot = one_hot.unsqueeze(1).repeat(1, vertice_num, 1)

        feat = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, one_hot], dim=2)


        if FLAGS.train:
            feat_face_re = f_global.view(bs, 1, f_global.shape[1]).repeat(1, feat.shape[1], 1).permute(0, 2, 1)


            conv1d_input = feat.permute(0, 2, 1)
            conv1d_out = self.conv1d_block(conv1d_input)

            recon = self.recon_head(conv1d_out)

            feat_face_in = torch.cat([feat_face_re, conv1d_out, vertices.permute(0, 2, 1)], dim=1)
            face = self.face_head(feat_face_in)
            return recon.permute(0, 2, 1), face.permute(0, 2, 1), feat
        else:
            recon, face = None, None

            return recon, face, feat



