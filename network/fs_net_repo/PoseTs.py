import torch.nn as nn
import torch
import torch.nn.functional as F
import absl.flags as flags
from absl import app

FLAGS = flags.FLAGS


class Pose_Ts(nn.Module):
    def __init__(self):
        super(Pose_Ts, self).__init__()
        self.f = FLAGS.feat_c_ts
        self.k = FLAGS.Ts_c

        self.conv1 = torch.nn.Conv1d(self.f, 1024, 1)

        self.conv2 = torch.nn.Conv1d(1024, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, self.k, 1)
        self.drop1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        x = self.leakyrelu(self.bn1(self.conv1(x)))
        x = self.leakyrelu(self.bn2(self.conv2(x)))

        x = torch.max(x, 2, keepdim=True)[0]

        x = self.leakyrelu(self.bn3(self.conv3(x)))
        x = self.drop1(x)
        x = self.conv4(x)

        x = x.squeeze(2)
        x = x.contiguous()
        xt = x[:, 0:3]
        xs = x[:, 3:6]
        return xt, xs

def main(argv):
    feature = torch.rand(3, 3, 1000)
    obj_id = torch.randint(low=0, high=15, size=[3, 1])
    net = Pose_Ts()
    out = net(feature, obj_id)
    t = 1

if __name__ == "__main__":
    app.run(main)
