import os
import re
import numpy as np
from absl import flags

FLAGS = flags.FLAGS

from tqdm import tqdm
import torch
import torch.utils.data as data


class PoseDataset(data.Dataset):
    def __init__(self, mode='train',test_mode='intra',n_pts=1024, img_size=256, per_obj=''):
        self.mode = mode
        self.n_pts = n_pts
        self.img_size = img_size
        self.voxel_size =  (1 / 100, 1 / 100, 1 / 100)

        self.gapart_train_size = {
            "Line_Fixed_Handle": 17015,
            "Round_Fixed_Handle": 1882,
            "Slider_Button": 82020,
            "Hinge_Door": 13077,
            "Slider_Drawer": 8406,
            "Slider_Lid": 757,
            "Hinge_Handle": 796,
            "Hinge_Lid": 2401,
            "Hinge_Knob": 3066,
        }
        self.gapart_test_size = {
            "Line_Fixed_Handle": {"intra": 4260, "inter": 2764},
            "Round_Fixed_Handle": {"intra": 457,  "inter": 1893},
            "Slider_Button": {"intra": 20504, "inter": 53707},
            "Hinge_Door": {"intra": 3262, "inter": 3481},
            "Slider_Drawer": {"intra": 2074, "inter": 4754},
            "Slider_Lid": {"intra": 215, "inter": 244},
            "Hinge_Handle": {"intra": 196, "inter": 320},
            "Hinge_Lid": {"intra": 629, "inter": 224},
            "Hinge_Knob": {"intra": 752, "inter": 1187},
        }
        self.gapart_name = ['Hinge_Knob','Hinge_Door','Slider_Button','Hinge_Knob','Line_Fixed_Handle','Round_Fixed_Handle','Slider_Drawer','Slider_Lid','Hinge_Lid','Hinge_Handle']
        self.gapart_name2id = {'Line_Fixed_Handle':1,'Round_Fixed_Handle':2,'Slider_Button':3,'Hinge_Door':4,'Slider_Drawer':5,'Slider_Lid':6,'Hinge_Lid':7,'Hinge_Knob':8,'Hinge_Handle':9}
        self.id2gapart_name = {1:'Line_Fixed_Handle',2:'Round_Fixed_Handle',3:'Slider_Button',4:'Hinge_Door',5:'Slider_Drawer',6:'Slider_Lid',7:'Hinge_Lid',8:'Hinge_Knob',9:'Hinge_Handle'}
        self.gapart = per_obj
        self.gapart_id = None

        if self.gapart in self.gapart_name:
            self.gapart_id = self.gapart_name2id[self.gapart]
            FLAGS.traindata_size = self.gapart_train_size[self.gapart]
            FLAGS.testdata_size_intra=self.gapart_test_size[self.gapart]['intra']
            FLAGS.testdata_size_inter=self.gapart_test_size[self.gapart]['inter']


        assert mode in ['train', 'test']
        if(mode=='train'):
            pose_file_path = f'{FLAGS.dataset_path}/pose/seen/train/{self.gapart_id}'
            gapart_file_path = f'{FLAGS.dataset_path}/sampled_gapart/seen/{self.gapart_id}/train'
            npcs_file_path = f'{FLAGS.dataset_path}/sampled_npcs/seen/{self.gapart_id}/train'

        elif(mode=='test'):
            assert test_mode in ['intra','inter']
            if(test_mode=='intra'):
                pose_file_path = f'{FLAGS.dataset_path}/pose/seen/test/{self.gapart_id}'
                gapart_file_path = f'{FLAGS.dataset_path}/sampled_gapart/seen/{self.gapart_id}/test'
                npcs_file_path = f'{FLAGS.dataset_path}/sampled_npcs/seen/{self.gapart_id}/test'

            else:
                pose_file_path = f'{FLAGS.dataset_path}/pose/unseen/{self.gapart_id}'
                gapart_file_path = f'{FLAGS.dataset_path}/sampled_gapart/unseen/{self.gapart_id}'
                npcs_file_path = f'{FLAGS.dataset_path}/sampled_npcs/unseen/{self.gapart_id}'

        print('>>>>>>>>>>>>>>>>>>>>>>>>  loading gapart and npcs pts !!!  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        if (mode == 'train'):
            progress_bar = tqdm(enumerate(sorted(os.listdir(gapart_file_path))), total=FLAGS.traindata_size)
            self.gapart_list = torch.zeros(FLAGS.traindata_size,FLAGS.n_points,3)
        else:
            if(test_mode=='intra'):
                progress_bar = tqdm(enumerate(sorted(os.listdir(gapart_file_path))), total=FLAGS.testdata_size_intra)
                self.gapart_list = torch.zeros(FLAGS.testdata_size_intra, FLAGS.n_points, 3)
            else:
                progress_bar = tqdm(enumerate(sorted(os.listdir(gapart_file_path))), total=FLAGS.testdata_size_inter)
                self.gapart_list = torch.zeros(FLAGS.testdata_size_inter, FLAGS.n_points, 3)
        if (mode == 'train'):
            progress_bar_npcs = tqdm(enumerate(sorted(os.listdir(npcs_file_path))), total=FLAGS.testdata_size_intra)
            self.npcs_list = torch.zeros(FLAGS.traindata_size,FLAGS.n_points,3)
        else:
            if(test_mode=='intra'):
                progress_bar_npcs = tqdm(enumerate(sorted(os.listdir(npcs_file_path))), total=FLAGS.testdata_size_intra)
                self.npcs_list = torch.zeros(FLAGS.testdata_size_intra, FLAGS.n_points, 3)
            else:
                progress_bar_npcs = tqdm(enumerate(sorted(os.listdir(npcs_file_path))), total=FLAGS.testdata_size_inter)
                self.npcs_list = torch.zeros(FLAGS.testdata_size_inter, FLAGS.n_points, 3)


        for i, filename in progress_bar:
            file_path = os.path.join(gapart_file_path, filename)
            content = torch.load(file_path).clone().detach()

            length = content.shape[0]
            self.gapart_list[i, :length] = content
            progress_bar.set_description(f"loading gapart {i + 1}")



        for i, filename in progress_bar_npcs:
            file_path = os.path.join(npcs_file_path, filename)
            content = torch.load(file_path).clone().detach()
            length = content.shape[0]
            self.npcs_list[i, :length] = content
            progress_bar_npcs.set_description(f"loading npcs {i + 1}")

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>  gapart/npcs pts are loaded ! <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        if(mode=='train'):
            self.pose_list = [0]*FLAGS.traindata_size
            self.index_list = torch.zeros(FLAGS.traindata_size)
        else:
            if(test_mode=='intra'):
                self.pose_list = [0] * FLAGS.testdata_size_intra
                self.index_list = torch.zeros(FLAGS.testdata_size_intra)
            else:
                self.pose_list = [0] * FLAGS.testdata_size_inter
                self.index_list = torch.zeros(FLAGS.testdata_size_inter)

        for i,filename in enumerate(sorted(os.listdir(pose_file_path))):
            file_path = os.path.join(pose_file_path, filename)
            content = torch.load(file_path)

            match = re.match(r'gapart_(\d+)_\d+.pth', filename)
            if match:
                self.index_list[i] = (int(match.group(1)))

            self.pose_list[i] = content

        if (mode == 'train'):
            self.rotation_list = [0] * FLAGS.traindata_size
            self.translation_list = [0] * FLAGS.traindata_size
            self.scale_list = [0] * FLAGS.traindata_size
        else:
            if(test_mode=='intra'):
                self.rotation_list = [0] * FLAGS.testdata_size_intra
                self.translation_list = [0] * FLAGS.testdata_size_intra
                self.scale_list = [0] * FLAGS.testdata_size_intra
            else:
                self.rotation_list = [0] * FLAGS.testdata_size_inter
                self.translation_list = [0] * FLAGS.testdata_size_inter
                self.scale_list = [0] * FLAGS.testdata_size_inter

        for i, pose in enumerate(self.pose_list):
            self.rotation_list[i] = torch.from_numpy(pose['rotation'])
            self.translation_list[i] = torch.from_numpy(pose['translation'])
            self.scale_list[i] = torch.from_numpy(pose['scale'])

        print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>  pose loading is finished !  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        length_gapart = len(self.gapart_list)
        length_pose = len(self.pose_list)
        length_R = len(self.rotation_list)
        length_T = len(self.translation_list)
        length_S = len(self.scale_list)
        assert length_gapart == length_pose == length_R == length_T == length_S,'data loading failed'
        mask = torch.eq(self.index_list,self.gapart_id)
        self.index_gapart = torch.nonzero(mask)
        self.length = len(self.index_gapart)
        print('{} gapart found.'.format(self.length))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if(idx>=self.length):
            return self.__getitem__((idx + 1) % self.__len__())
        data_dict = {}
        index = self.index_gapart[idx]
        rotation = self.rotation_list[index].float()
        translation = self.translation_list[index].squeeze().float()
        if(torch.isnan(translation).any()):
            return self.__getitem__((idx + 1) % self.__len__())
        scale = self.scale_list[index].float()
        pts = self.gapart_list[index].squeeze().float()
        npcs = self.npcs_list[index].squeeze().float()

        gapart_id = self.gapart_id
        mean_shape = self.get_mean_shape(self.id2gapart_name[self.gapart_id])
        sym_info = self.get_sym_info(self.id2gapart_name[self.gapart_id])
        mean_trans = pts.mean(dim=0)
        mean_shape = mean_shape / 1000

        data_dict['pts'] = pts.clone().detach().contiguous()
        data_dict['rotation'] = rotation.clone().detach().contiguous()
        data_dict['translation'] = translation.clone().detach().contiguous()
        data_dict['scale'] = scale.clone().detach().contiguous()
        data_dict['id'] = torch.tensor(gapart_id, dtype=torch.float32).contiguous()
        data_dict['mean_shape'] = torch.tensor(mean_shape,dtype=torch.float32).contiguous()
        data_dict['sym_info'] = torch.tensor(sym_info, dtype=torch.int32).contiguous()
        data_dict['mean_trans'] = mean_trans.clone().detach().contiguous()
        data_dict['npcs'] = npcs.clone().detach().contiguous()

        return data_dict


    def get_mean_shape(self,c):
        if c == 'Round_Fixed_Handle':
            unitx = 87
            unity = 220
            unitz = 89
        elif c == 'Slider_Drawer':
            unitx = 165
            unity = 80
            unitz = 165
        elif c == 'Slider_Lid':
            unitx = 88
            unity = 128
            unitz = 156
        elif c == 'Hinge_Lid':
            unitx = 68
            unity = 146
            unitz = 72
        elif c == 'Hinge_Handle':
            unitx = 346
            unity = 200
            unitz = 335
        elif c == 'Line_Fixed_Handle':
            unitx = 92
            unity = 110
            unitz = 70
        elif c == 'Hinge_Knob':
            unitx = 60
            unity = 55
            unitz = 35
        elif c == 'Slider_Button':
            unitx = 60
            unity = 55
            unitz = 35
        elif c == 'Hinge_Door':
            unitx = 60
            unity = 55
            unitz = 35
        elif c == 'Hinge_Knob':
            unitx = 60
            unity = 55
            unitz = 35
        else:
            unitx = 0
            unity = 0
            unitz = 0
            print('This category is not recorded in my little brain.')
            raise NotImplementedError
        return np.array([unitx,unity,unitz])

    def get_sym_info(self, c):
        """
        sym_info:
            0: no symmetry
            1: axis symmetry
            2: reflection symmetry
        """
        if c == 'Round_Fixed_Handle':
            sym = 1
        elif c == 'Slider_Drawer':
            sym = 0
        elif c == 'Slider_Lid':
            sym = 1
        elif c == 'Hinge_Lid':
            sym = 2
        elif c == 'Hinge_Handle':
            sym = 2
        elif c == 'Line_Fixed_Handle':
            sym = 2
        elif c == 'Hinge_Knob':
            sym = 1
        elif c == 'Slider_Button':
            sym = 1
        elif c == 'Hinge_Door':
            sym = 2
        else:
            raise ValueError(f"Unknown category: {c}")

        return np.int32(sym)
