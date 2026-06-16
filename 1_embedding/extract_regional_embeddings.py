from functools import reduce
import glob
from functools import partial
import multiprocessing
from monai.utils.type_conversion import convert_data_type
from monai.utils.enums import TransformBackends
from monai.utils import (
    Method,
    NumpyPadMode,
    PytorchPadMode,
    ensure_tuple,
    ensure_tuple_rep,
    fall_back_tuple,
    look_up_option,
)
from monai.transforms.utils import (
    compute_divisible_spatial_size,
    convert_pad_mode,
    # we need this for customized implementation for only to sample from the foreground
    correct_crop_centers,
    generate_label_classes_crop_centers,
    generate_pos_neg_label_crop_centers,
    generate_spatial_bounding_box,
    is_positive,
    map_binary_to_indices,
    map_classes_to_indices,
    weighted_patch_samples,
)
from monai.transforms.transform import Randomizable, Transform
from monai.data.utils import get_random_patch, get_valid_patch_size
from monai.config.type_definitions import NdarrayOrTensor
from monai.config import IndexSelection
from torch.nn.functional import pad as pad_pt
from math import ceil
from itertools import chain
from monai.transforms.utils_pytorch_numpy_unification import floor_divide, maximum
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
import monai
from GPUtil import showUtilization as gpu_usage
import time
import numpy as np
import random
import itertools
import matplotlib.pyplot as plt
import shutil

import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from scipy.ndimage import distance_transform_edt

import os
import nibabel as nib
from zipfile import ZipFile
import random

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
from paths import cfg

try:

    # from . import config
    # from . import models
    from . import config_pixpro as config
    from . import pixpro2
    from . import config_loader_pixpro
except:

    # import config_pixpro
    # import config
    # import models
    import config_pixpro as config
    import pixpro2
    import config_loader_pixpro

# from . import gradient_check
################# Region dictionary###############
region_dict = {1: 'Left Cerebral White Matter',
               2: 'Left Cerebral Cortex',
               3:  'Left Lateral Ventricle',
               4:  'Left Thalamus',
               5:  'Left Caudate',
               6:  'Left Putamen',
               7:  'Left Pallidum',
               8:  'Brain-Stem',
               9: 'Left Hippocampus',
               10: 'Left Amygdala',
               11: 'Left Accumbens',
               12: 'Right Cerebral White Matter',
               13: 'Right Cerebral Cortex',
               14: 'Right Lateral Ventricle',
               15: 'Right Thalamus',
               16: 'Right Caudate',
               17: 'Right Putamen',
               18: 'Right Pallidum',
               19: 'Right Hippocampus',
               20: 'Right Amygdala',
               21: 'Right Accumbens'}


# first_regional dictionary
# https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIRST/UserGuide#Labels
# https://biobank.ctsu.ox.ac.uk/crystal/crystal/docs/brain_mri.pdf (for label id of CSF, GM, WM), page#13
first_region_dict = {


    10: 'Left_Thalamus_Proper',
    11: 'Left_Caudate',
    12: 'Left_Putamen',
    13: 'Left_Pallidum',
    16: 'Brain_Stem _or_4th_Ventricle',
    17: 'Left_Hippocampus',
    18: 'Left_Amygdala',
    26: 'Left_Accumbens-area',
    49: 'Right_Thalamus-Proper',
    50: 'Right_Caudate',
    51: 'Right_Putamen',
    52: 'Right_Pallidum',
    53: 'Right_Hippocampus',
    54: 'Right_Amygdala',
    58: 'Right_Accumbens-area'
}

brain_seg_dict = {
    1: 'CSF',
    2: 'GM',
    3: 'WM'

}

region_dict_lin_min = {0: first_region_dict, 1: brain_seg_dict}


def save_region_dict(save_path):
    with open(save_path, 'wb') as f:
        pkl.dump(region_dict_lin_min, f)

######################## Data Processing ###########################


########################### Start from here for Foreground image pipeline##########


brain_seg_dict = {
    0: 'CSF',
    1: 'GM',
    2: 'WM'

}

first_region_dict = {


    10: 'Left_Thalamus_Proper',
    11: 'Left_Caudate',
    12: 'Left_Putamen',
    13: 'Left_Pallidum',
    16: 'Brain_Stem _or_4th_Ventricle',
    17: 'Left_Hippocampus',
    18: 'Left_Amygdala',
    26: 'Left_Accumbens-area',
    49: 'Right_Thalamus-Proper',
    50: 'Right_Caudate',
    51: 'Right_Putamen',
    52: 'Right_Pallidum',
    53: 'Right_Hippocampus',
    54: 'Right_Amygdala',
    58: 'Right_Accumbens-area'
}


# we have to keep this sequence, because in the masks from the dataloader, index 0 is for brain_segmentation
region_dict_lin_min = {0: brain_seg_dict, 1: first_region_dict}


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, gpu_id, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.gpu_id = gpu_id

    def forward(self, x):
        x = x  # .to(self.gpu_id)
        residual = x
        # print(f"x: {x.shape}")
        # print(f"x.device {x.device}")
        # print(f"self.conv1.device{self.conv1.weight.device}")
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print(f"out: {out.shape}")
        if self.downsample is not None:
            residual = self.downsample(x)
        # if self.downsample is not None:
            # print(f"downsample is not none")
            # print(f"residual: {residual.shape}")
        # else:
            # print(f"No downsampling  is done")
            # print(f"x: {x.shape}")
        # print("========")
        out += residual
        out = self.relu(out)
        # print("done one block")
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Resnet18(nn.Module):
    def __init__(self, block, block_inplanes, layers, n_input_channels=1, conv1_t_size=3, conv1_t_stride=1, no_max_pool=True, shortcut_type='B', gpu_list=[]):
        super(Resnet18, self).__init__()
        self.block = block
        self.planes = 64
        self.blocks = 2
        self.stride = 1
        self.in_planes = 64
        self.gpu_list = gpu_list
        self.layer_gpu_mpping = {

            "c1": gpu_list[0],
            "c2": gpu_list[1],
            "c3": gpu_list[1],
            "c4": gpu_list[-1],
            "c5": gpu_list[-1]


        }
        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(7, conv1_t_size, 7),
                               stride=(1, conv1_t_stride, 1),
                               padding=(3, conv1_t_size // 2, 3),
                               bias=False)  # .to(self.layer_gpu_mpping["c1"])
        # self.conv1.to(self.layer_gpu_mpping["c1"])
        # self.conv2_x = self._make_layer(self.block, self.planes, self.blocks, self.stride)
        self.conv2_x = self._make_layer(block,
                                        block_inplanes[0],
                                        layers[0],
                                        shortcut_type,
                                        gpu_id=self.layer_gpu_mpping["c2"],
                                        conv_type="B",
                                        stride=2,
                                        apply_z_stride=2

                                        )
        # self.conv2_x.to(self.layer_gpu_mpping["c2"])

        self.conv3_x = self._make_layer(block,
                                        block_inplanes[1],
                                        layers[1],
                                        shortcut_type,
                                        gpu_id=self.layer_gpu_mpping["c3"],
                                        conv_type="B",

                                        stride=2,
                                        apply_z_stride=1

                                        )
        # self.conv3_x.to(self.layer_gpu_mpping["c3"])
        self.conv4_x = self._make_layer(block,
                                        block_inplanes[2],
                                        layers[2],
                                        shortcut_type,
                                        gpu_id=self.layer_gpu_mpping["c4"],
                                        conv_type="B",

                                        stride=2,
                                        apply_z_stride=1

                                        )
        # self.conv4_x.to(self.layer_gpu_mpping["c4"])
        # device, = list(set(p.device for p in self.conv4_x.parameters()))
        # print()
        # print(f"device: cov4_x 's device is {device}")
        # print(f"using alternative method, self.conv4_x.weight.device:{next(self.conv4_x.parameters()).device}")

        self.conv5_x = self._make_layer(block,
                                        block_inplanes[3],
                                        layers[3],
                                        shortcut_type,
                                        gpu_id=self.layer_gpu_mpping["c5"],
                                        conv_type="B",

                                        stride=2,
                                        apply_z_stride=2

                                        )
        self.already_sent = True

        # self.conv5_x.to(self.layer_gpu_mpping["c5"])
    def _make_layer(self, block, planes, blocks, shortcut_type, gpu_id, conv_type="A", stride=1, apply_z_stride=1):
        downsample = None

        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=1)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes *
                              block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        if conv_type == "A":
            layers.append(
                block(in_planes=self.in_planes,
                      planes=planes,
                      stride=stride,
                      downsample=downsample,
                      gpu_id=gpu_id
                      ))
        else:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion,
                          [stride, stride, apply_z_stride]),
                nn.BatchNorm3d(planes * block.expansion)).to(gpu_id)
            layers.append(
                block(in_planes=self.in_planes,
                      planes=planes,
                      stride=[stride, stride, apply_z_stride],
                      downsample=downsample,
                      gpu_id=gpu_id
                      ))

        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes,
                                gpu_id=gpu_id
                                ))

        return nn.Sequential(*layers)  # .to(gpu_id)

    def send(self):
        if not self.already_sent:
            self.conv1.to(self.layer_gpu_mpping["c1"])
            self.conv2_x.to(self.layer_gpu_mpping["c2"])
            self.conv3_x.to(self.layer_gpu_mpping["c3"])
            self.conv4_x.to(self.layer_gpu_mpping["c4"])
            self.conv5_x.to(self.layer_gpu_mpping["c5"])

    def forward(self, x):

        if not self.already_sent:
            self.send()
        # x = [32,96,96] or x = [a*32, b*32, c*32] , if x = [32,96,96] then a = 1, b = 3,c=3
        # print(f"computing through self.conv1")
        '''
        if x.device !=self.layer_gpu_mpping["c1"]:
            x = x.to(self.layer_gpu_mpping["c1"])
        '''
        x = self.conv1(x)  # [32,96,96] =>  [a*32, b*32, c*32]

        # print(f"computing through self.conv2")
        '''
        if x.device !=self.layer_gpu_mpping["c2"]:
             x = x.to(self.layer_gpu_mpping["c2"])
        '''
        c2 = self.conv2_x(x)  # [16,48,48] => [a*32/2, b*32//2, c*32//2]

        # print(f"computing through self.conv3")
        '''
        if c2.device !=self.layer_gpu_mpping["c3"]:
             c2 = c2.to(self.layer_gpu_mpping["c3"])
        '''
        c3 = self.conv3_x(c2)  # [16,24,24] => [a*32/2, b*32//4, c*32//4]

        # print(f"computing through self.conv4")
        '''
        if c3.device !=self.layer_gpu_mpping["c4"]:
             c3 = c3.to(self.layer_gpu_mpping["c4"])
        '''
        # print(f"c3's device :{c3.device}")
        # self.conv4_x.to(self.layer_gpu_mpping["c4"])
        # print(f"self.conv4_x.weight.device:{next(self.conv4_x.parameters()).device}")

        c4 = self.conv4_x(c3)  # [16,12,12] => [a*32/2, b*32//8, c*32//8]

        '''
        if c4.device !=self.layer_gpu_mpping["c5"]:
            c4 =  c4.to(self.layer_gpu_mpping["c5"])
        #print(f"computing through self.conv5")
        #self.conv5_x.to(self.layer_gpu_mpping["c5"])
        '''
        c5 = self.conv5_x(c4)  # [8,6,6] => [a*32/4, b*32//16, c*32//16]
        # print("DOne computing c5")
        return (c2, c3, c4, c5)


def generate_model(gpu_list):
    block = BasicBlock
    block_inplanes = [64, 128, 256, 512]
    layers = [2, 2, 2, 2]
    return Resnet18(block=BasicBlock, block_inplanes=block_inplanes, layers=layers, gpu_list=gpu_list)


class FPN(nn.Module):
    def __init__(self, gpu_list):
        super().__init__()
        model_depth = 18
        self.backbone_resnet = generate_model(
            gpu_list)  # return tensors in gpu_3
        self.gpu = gpu_list[-1]

        # Smooth layers
        self.smooth2 = conv3x3x3(128, 128)  # .to(self.gpu)
        self.smooth5 = conv3x3x3(128, 128)  # .to(self.gpu)

        # Lateral layers

        self.latlayer5 = conv1x1x1(512, 128)  # .to(self.gpu)
        self.latlayer4 = conv1x1x1(256, 128)  # .to(self.gpu)
        self.latlayer3 = conv1x1x1(128, 128)  # .to(self.gpu)
        self.latlayer2 = conv1x1x1(64, 128)  # .to(self.gpu)

        self.already_sent = True

    def send(self):
        if not self.already_sent:
            self.smooth2.to(self.gpu)
            self.smooth5.to(self.gpu)
            self.latlayer5.to(self.gpu)
            self.latlayer4.to(self.gpu)
            self.latlayer3.to(self.gpu)
            self.latlayer2.to(self.gpu)
            self.already_sent = True

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose trilinear upsample which supports arbitrary output sizes, because it expects 5D tensor.
        '''
        _, _, D, H, W = y.size()
        return F.upsample(x, size=(D, H, W), mode='trilinear') + y

    def forward(self, x):
        if not self.already_sent:
            self.send()

        # print(f"x: {x.shape}")
        b, c, img_D, img_W, img_H = x.shape
        # top down flow
        c2, c3, c4, c5 = self.backbone_resnet(x)

        # c2,c3,c4,c5 = [z.to(self.gpu) for z in [c2,c3,c4,c5] ]

        # print(f"c2,c3,c4,c5:{[i.device for i in [c2,c3,c4,c5]]}")
        '''
        print("before applying lateral connection")
        
        print(f"c2 => {c2.shape}")
        print(f"c3 => {c3.shape}")
        print(f"c4 => {c4.shape}")
        print(f"c5 => {c5.shape}")
        print()
        '''
        # lateral connection
        # self.latlayer5.to(self.gpu)
        c5 = self.latlayer5(c5)

        # self.latlayer4.to(self.gpu)
        c4 = self.latlayer4(c4)

        # self.latlayer3.to(self.gpu)
        c3 = self.latlayer3(c3)

        # self.latlayer2.to(self.gpu)
        c2 = self.latlayer2(c2)

        '''
        print("after applying lateral connection")
        
        print(f"c2 => {c2.shape}")
        print(f"c3 => {c3.shape}")
        print(f"c4 => {c4.shape}")
        print(f"c5 => {c5.shape}")      
        print()
        '''
        # print(f"after lateral connection :c5,c4,c3,c2 {c5.shape,c4.shape,c3.shape,c2.shape }")

        # upsample

        p3 = self._upsample_add(c4, c3)
        # print(f"p3: c4+c3 => {p3.shape}")
        p2 = self._upsample_add(p3, c2)
        # print(f"p2: p3+c2 => {p2.shape}")
        '''
        print("after applying lateral upsamplingconnection")
        
        print(f"c4+c3 => {p3.shape}")
        print(f"p3+c2 => {p2.shape}")
        
        print()
        '''
        # smooth
        # self.smooth5.to(self.gpu)
        global_embedding = self.smooth5(c5)
        # print(f"global_embedding => {global_embedding.shape}")
        global_embedding = F.normalize(global_embedding, p=2, dim=(2, 3, 4))
        # print(f"global_embedding after L2 normalization => {global_embedding.shape}")

        # self.smooth2.to(self.gpu)
        local_embedding = self.smooth2(p2)
        # print(f"local_embedding => {local_embedding.shape}")
        local_embedding = F.normalize(local_embedding, p=2, dim=(2, 3, 4))
        # print(f"local_embedding after L2 normalization => {local_embedding.shape}")

        '''
        #upsample global_embedding to have same shape as local_embedding
        _,_,D,H,W = local_embedding.shape
        global_embedding = F.upsample(global_embedding, size=(D,H,W), mode='trilinear')
        #print(f"global_embedding after upsampled => {global_embedding.shape}")   
        
        
        ## upsample both to patch level
        
        local_embedding = F.upsample(local_embedding, size=(img_D, img_W,img_H), mode='trilinear')
        global_embedding = F.upsample(global_embedding, size=(img_D, img_W,img_H), mode='trilinear')
        '''
        return local_embedding, global_embedding  # this two are in the gpu[-1]s.


class Model(torch.nn.Module):
    def __init__(self, n_batch, n_pos_voxel, global_n_neg, local_cand_neg,
                 embedding_dim, global_n_random, los_temp, device, radius=0, gpu_list=[]):
        super().__init__()
        self.fpn = FPN(gpu_list)
        self.n_batch = n_batch
        self.n_pos_voxel = n_pos_voxel
        self.global_n_neg = global_n_neg
        self.local_cand_neg = local_cand_neg
        self.embedding_dim = embedding_dim
        self.global_n_random = global_n_random
        self.los_temp = los_temp
        self.local_softmax = torch.nn.Softmax(dim=-1)
        self.global_softmax = torch.nn.Softmax(dim=-1)
        self.device = device
        self.radius = radius
        self.gpu_list = gpu_list

    def forward(self, patch_ls, pos, is_test_mode=False):
        '''
        return ([local_embedding_1st_ls ,global_embedding_1st_ls],
            [local_embedding_2nd_ls ,global_embedding_2nd_ls],[local_loss,global_loss])
        pos should be in gpu_3
        '''
        print(f'models forward is called')
        # step 1
        # 1st achor
        n_batch = patch_ls[0].shape[0]
        device = patch_ls[0].device
        local_embed, global_embed = self.fpn(torch.cat(patch_ls, dim=0))
        local_embed, global_embed = local_embed.to(
            self.device), global_embed.to(self.device)
        # print(f"after calcuting local, global embedding")
        # gpu_usage()
        # normalize over the channel dimension
        local_embed, global_embed = F.normalize(
            local_embed, p=2.0, dim=1), F.normalize(global_embed, p=2.0, dim=1)

        local_embedding_1st_ls, global_embedding_1st_ls = local_embed[
            :n_batch], global_embed[:n_batch]
        local_embedding_2nd_ls, global_embedding_2nd_ls = local_embed[
            n_batch:], global_embed[n_batch:]

        '''
        local_embedding_1st_ls ,global_embedding_1st_ls = self.fpn(patch_ls[0]) #torch.Size([n_batch, self.embedding_dim, dim_z, dim_x, dim_y])

        #2nd achors 
        local_embedding_2nd_ls ,global_embedding_2nd_ls = self.fpn(patch_ls[1])
        '''
        # pos = pos.to(self.gpu_list[-1])
        pos = pos.view(n_batch, self.n_pos_voxel, -1).to(dtype=torch.long)

        # step 5: extract positive pair embedding
        b, c, w, h, d = local_embedding_2nd_ls.shape  # x,y,z

        # we need to divide the coordinates of to z/4 axis and x,y axis by 16 (global embeddings)
        # each last dimension in pos_z contains the coordinates of achor  coordinates : (0,1,2)th indexes, and the paired one is (3,4,5)th indexes
        # z dimension is the last dimension. so z dimension would 2,5th idexes
        pos_z = pos.clone()
        pos_z[:, :, [2, 5]] = (
            pos_z[:, :, [2, 5]]//torch.tensor([4], dtype=torch.long).to(device)).to(torch.long)
        pos_z[:, :, [0, 1, 3, 4]] = (pos_z[:, :, [
                                     0, 1, 3, 4]]//torch.tensor([16], dtype=torch.long).to(device)).to(torch.long)

        # we need to divide the all coordinates by 2 ( local embedding)
        pos_xy = pos.clone()
        pos_xy = (
            pos_xy//torch.tensor([2], dtype=torch.long).to(device)).to(torch.long)

        # added in this version
        '''
        we will make the similarity of voxels within the 'self.radius' to -inf
        '''
        left_pos_x, left_pos_y, left_pos_z = pos[:, :, [
            0, 0+3]]-self.radius, pos[:, :, [1, 1+3]]-self.radius, pos[:, :, [2, 2+3]]-self.radius
        right_pos_x, right_pos_y, right_pos_z = pos[:, :, [
            0, 0+3]]+self.radius, pos[:, :, [1, 1+3]]+self.radius, pos[:, :, [2, 2+3]]+self.radius

        left_pos_x, left_pos_y, left_pos_z = torch.where(left_pos_x < 0, 0, left_pos_x), torch.where(
            left_pos_y < 0, 0, left_pos_y), torch.where(left_pos_z < 0, 0, left_pos_z)
        right_pos_x, right_pos_y, right_pos_z = torch.where(right_pos_x >= w, w-1, right_pos_x), torch.where(
            right_pos_y >= h, h-1, right_pos_y), torch.where(right_pos_z >= d, d-1, right_pos_z)

        # now scale down to match the scaling done in the network
        # local embedding
        left_pos_x = left_pos_x//torch.tensor([2],
                                              dtype=torch.long).to(device).to(torch.long)
        left_pos_z = left_pos_z//torch.tensor([2],
                                              dtype=torch.long).to(device).to(torch.long)
        left_pos_y = left_pos_y//torch.tensor([2],
                                              dtype=torch.long).to(device).to(torch.long)

        right_pos_x = right_pos_x//torch.tensor(
            [2], dtype=torch.long).to(device).to(torch.long)
        right_pos_z = right_pos_z//torch.tensor(
            [2], dtype=torch.long).to(device).to(torch.long)
        right_pos_y = right_pos_y//torch.tensor(
            [2], dtype=torch.long).to(device).to(torch.long)

        # left_pos_x,left_pos_z,left_pos_y = [(i//torch.tensor([2],dtype=torch.long)).to(device)).to(torch.long) for i in [left_pos_x,left_pos_z,left_pos_y]]
        # right_pos_x,right_pos_z,right_pos_y = [(i//torch.tensor([2],dtype=torch.long)).to(device)).to(torch.long) for i in [right_pos_x,right_pos_z,right_pos_y]]

        # end of this newly added version

        # now extract embedding

        # global
        anchor_global_1st_ls = global_embedding_1st_ls[torch.arange(n_batch).reshape(n_batch, -1).long(), :, pos_z[torch.arange(
            n_batch).long(), :, 0], pos_z[torch.arange(n_batch).long(), :, 1], pos_z[torch.arange(n_batch).long(), :, 2]]
        anchor_global_2nd_ls = global_embedding_2nd_ls[torch.arange(n_batch).reshape(n_batch, -1).long(), :, pos_z[torch.arange(
            n_batch).long(), :, 3], pos_z[torch.arange(n_batch).long(), :, 4], pos_z[torch.arange(n_batch).long(), :, 5]]
        # local
        anchor_local_1st_ls = local_embedding_1st_ls[torch.arange(n_batch).reshape(n_batch, -1).long(), :, pos_xy[torch.arange(
            n_batch).long(), :, 0], pos_xy[torch.arange(n_batch).long(), :, 1], pos_xy[torch.arange(n_batch).long(), :, 2]]
        anchor_local_2nd_ls = local_embedding_2nd_ls[torch.arange(n_batch).reshape(n_batch, -1).long(), :, pos_xy[torch.arange(
            n_batch).long(), :, 3], pos_xy[torch.arange(n_batch).long(), :, 4], pos_xy[torch.arange(n_batch).long(), :, 5]]
        # print(f"anchor_local_1st_ls: {anchor_local_1st_ls[:,:,:100]}")
        # print(f"anchor_local_2nd_ls: {anchor_local_2nd_ls[:,:,:100]}")

        # step 7: calculate similarity maps sg, sg',sl,sl'

        # sg would be the similarity map of anchor_global_1st_ls with global_embedding_1st_ls
        sg = torch.einsum(
            'bij, bjklm-> biklm', [anchor_global_1st_ls, global_embedding_1st_ls]).to(device)
        # sg' would be the similarity map of anchor_global_1st_ls with global_embedding_2nd_ls
        sg_prime = torch.einsum(
            'bij, bjklm-> biklm', [anchor_global_1st_ls, global_embedding_2nd_ls]).to(device)

        # lg would be the similarity map of anchor_local_1st_ls with local_embedding_1st_ls
        sl = torch.einsum(
            'bij, bjklm-> biklm', [anchor_local_1st_ls, local_embedding_1st_ls]).to(device)
        # sg' would be the similarity map of anchor_global_1st_ls with global_embedding_2nd_ls
        sl_prime = torch.einsum(
            'bij, bjklm-> biklm', [anchor_local_1st_ls, local_embedding_2nd_ls]).to(device)

        # print(f"after calcuting local similarity, global similarity")
        # gpu_usage()

        # want to see which voxels are most similar to the positive voxels
        # debug dump path; set via config
        with open("<EXTERNAL: debug similarity dump path (s.pkl)>", "wb") as f:
            pkl.dump([pos, pos_xy, sl, sl_prime], f)

        # sg.shape,sg_prime.shape, sl.shape,sl_prime.shape -> (n_batch,self.n_pos_voxel,dim_z,dim_x,dim_y)

        # saving sg,sg prime for finding local hard embeddings
        sl_x, sl_z, sl_y = sl.size()[-3:]
        sg_local = sg.clone()
        sg_prime_local = sg_prime.clone()
        sg_local[torch.arange(n_batch).reshape(n_batch, -1).long().to(device), torch.arange(self.n_pos_voxel).long().to(device), pos_z[torch.arange(n_batch).long().to(
            device), :, 0], pos_z[torch.arange(n_batch).long().to(device), :, 1], pos_z[torch.arange(n_batch).long().to(device), :, 2]] = torch.tensor(-float("inf")).to(device)

        sg_prime_local[torch.arange(n_batch).reshape(n_batch, -1).long().to(device), torch.arange(self.n_pos_voxel).long().to(device), pos_z[torch.arange(n_batch).long().to(
            device), :, 3], pos_z[torch.arange(n_batch).long().to(device), :, 4], pos_z[torch.arange(n_batch).long().to(device), :, 5]] = torch.tensor(-float("inf")).to(device)

        sg_local = F.upsample(sg_local, size=(
            sl_x, sl_z, sl_y), mode='trilinear')
        sg_prime_local = F.upsample(sg_prime_local, size=(
            sl_x, sl_z, sl_y), mode='trilinear')

        # step 8: Finiding global hard negatives

        # set similarity of positive voxel with themselves to zero
        sg[torch.arange(n_batch).reshape(n_batch, -1).long().to(device), torch.arange(self.n_pos_voxel).long().to(device), pos_z[torch.arange(n_batch).long().to(device),
                                                                                                                                 :, 0], pos_z[torch.arange(n_batch).long().to(device), :, 1], pos_z[torch.arange(n_batch).long().to(device), :, 2]] = torch.tensor(-float("inf")).to(device)

        sg_prime[torch.arange(n_batch).reshape(n_batch, -1).long().to(device), torch.arange(self.n_pos_voxel).long().to(device), pos_z[torch.arange(n_batch).long().to(
            device), :, 3], pos_z[torch.arange(n_batch).long().to(device), :, 4], pos_z[torch.arange(n_batch).long().to(device), :, 5]] = torch.tensor(-float("inf")).to(device)

        # sort the sg,sg_prime, sl,sl_prime
        sg = sg.view(b, self.n_pos_voxel, -1)
        sg_prime = sg_prime.view(b, self.n_pos_voxel, -1)
        _, indices_g = torch.sort(sg, descending=True)
        _, indices_g_prime = torch.sort(sg_prime, descending=True)

        indices_g = indices_g[..., :self.global_n_neg]
        indices_g_prime = indices_g_prime[..., :self.global_n_neg]
        # indices_g.shape -> torch.Size([n_batch, self.self.n_pos_voxel, global_n_neg])

        global_embedding_1st_ls_flat = global_embedding_1st_ls.reshape(
            n_batch, self.embedding_dim, -1)
        global_embedding_2nd_ls_flat = global_embedding_2nd_ls.reshape(
            n_batch, self.embedding_dim, -1)
        # global_embedding_1st_ls_flat.shape, global_embedding_2nd_ls_flat.shape
        # global_embedding_1st_ls_flat.shape -> (n_batch,self.self.n_pos_voxel,self.self.embedding_dim,dim_x,dim_z*dim_y)

        global_embedding_1st_ls_flat = global_embedding_1st_ls_flat.unsqueeze(
            1).expand(-1, self.n_pos_voxel, -1, -1)
        global_embedding_2nd_ls_flat = global_embedding_2nd_ls_flat.unsqueeze(
            1).expand(-1, self.n_pos_voxel, -1, -1)

        # our target global hard negative sample should be [3,4,128,10]
        indices_g = indices_g.unsqueeze(
            2).expand(-1, -1, self.embedding_dim, -1)
        # (n_batch,self.n_pos_voxel,self.embedding_dim,self.global_n_neg)
        indices_g_prime = indices_g_prime.unsqueeze(
            2).expand(-1, -1, self.embedding_dim, -1)

        neg_global_emb_g = torch.gather(
            global_embedding_1st_ls_flat, dim=-1, index=indices_g)
        neg_global_emb_g_prime = torch.gather(
            global_embedding_2nd_ls_flat, dim=-1, index=indices_g_prime)

        del global_embedding_1st_ls_flat
        del global_embedding_2nd_ls_flat
        # neg_global_emb_g_prime -> (n_batch,self.n_pos_voxel,self.self.embedding_dim,self.global_n_neg)

        # concat global hard embedding from sg,sg_prime and randomly choose self_global_n_neg
        neg_global_emb = torch.cat(
            [neg_global_emb_g, neg_global_emb_g_prime], dim=-1)

        del neg_global_emb_g
        del neg_global_emb_g_prime
        # neg_global_emb.shape -> ((n_batch,self.n_pos_voxel,self.self.embedding_dim,2*self.global_n_neg))
        # indices = torch.empty(self.global_n_neg,dtype = torch.long).random_(0,2*self.global_n_neg)
        neg_glbal_f_embed = torch.index_select(neg_global_emb, dim=-1, index=(
            torch.randperm(2*self.global_n_neg)[:self.global_n_neg]).to(device))
        # neg_glbal_f_embed -> (n_batch,self.n_pos_voxel,self.self.embedding_dim,self.global_n_neg)
        # print(f"after calcuting global hard negatives")
        # gpu_usage()
        # step: 10 now select n_cand hard local embeddings
        sl = sl+sg_local
        sl_prime = sl_prime + sg_prime_local

        # instead of masking only the positive voxel, mask all the voxels in the radius

        # sl[torch.arange(n_batch).reshape(n_batch,-1).long().to(device),torch.arange(self.n_pos_voxel).long().to(device),pos_xy[torch.arange(n_batch).long().to(device),:,0],pos_xy[torch.arange(n_batch).long().to(device),:,1],pos_xy[torch.arange(n_batch).long().to(device),:,2]] = torch.tensor(-float("inf")).to(device)

        # sl_prime[torch.arange(n_batch).reshape(n_batch,-1).long().to(device),torch.arange(self.n_pos_voxel).long().to(device),pos_xy[torch.arange(n_batch).long().to(device),:,3],pos_xy[torch.arange(n_batch).long().to(device),:,4],pos_xy[torch.arange(n_batch).long().to(device),:,5]] = torch.tensor(-float("inf")).to(device)
        for ib in range(n_batch):
            for ip in range(self.n_pos_voxel):
                # sl
                sl[ib, ip, left_pos_x[ib, ip, 0].item():right_pos_x[ib, ip, 0].item(),
                   left_pos_y[ib, ip, 0].item():right_pos_y[ib, ip, 0].item(),
                   left_pos_z[ib, ip, 0].item():right_pos_z[ib, ip, 0].item()] = torch.tensor(-float("inf"))

                # sl_prime
                sl_prime[ib, ip, left_pos_x[ib, ip, 1].item():right_pos_x[ib, ip, 1].item(),
                         left_pos_y[ib, ip, 1].item():right_pos_y[ib, ip, 1].item(),
                         left_pos_z[ib, ip, 1].item():right_pos_z[ib, ip, 1].item()] = torch.tensor(-float("inf"))

        sl = sl.view(b, self.n_pos_voxel, -1)
        sl_prime = sl_prime.view(b, self.n_pos_voxel, -1)

        _, indices_sgl = torch.sort(sl, descending=True)
        _, indices_sgl_prime = torch.sort(sl_prime, descending=True)

        del sl
        del sl_prime

        del sg
        del sg_prime

        indices_sgl = indices_sgl[..., :self.local_cand_neg]
        indices_sgl_prime = indices_sgl_prime[..., :self.local_cand_neg]

        # indices_sgl.shape -> (n_batch,self.self.n_pos_voxel,self.local_cand_neg)
        local_embedding_1st_ls_flat = local_embedding_1st_ls.reshape(
            b, self.embedding_dim, -1)
        local_embedding_2nd_ls_flat = local_embedding_2nd_ls.reshape(
            b, self.embedding_dim, -1)

        local_embedding_1st_ls_flat = local_embedding_1st_ls_flat.unsqueeze(
            1).expand(-1, self.n_pos_voxel, -1, -1)
        local_embedding_2nd_ls_flat = local_embedding_2nd_ls_flat.unsqueeze(
            1).expand(-1, self.n_pos_voxel, -1, -1)

        # local_embedding_1st_ls_flat.shape -> (n_batch,self.n_pos_voxel,self.self.embedding_dim,dim_z*dim_x*dim_y)

        indices_sgl = indices_sgl.unsqueeze(
            2).expand(-1, -1, self.embedding_dim, -1)
        indices_sgl_prime = indices_sgl_prime.unsqueeze(
            2).expand(-1, -1, self.embedding_dim, -1)
        # indices_sgl_prime -> (n_batch,self.n_pos_voxel,self.self.embedding_dim,self.local_cand_neg)

        neg_local_emb_l = torch.gather(
            local_embedding_1st_ls_flat, dim=-1, index=indices_sgl)
        neg_local_emb_l_prime = torch.gather(
            local_embedding_2nd_ls_flat, dim=-1, index=indices_sgl_prime)
        del local_embedding_2nd_ls_flat
        del local_embedding_1st_ls_flat
        # concate local_hard_negative from sl,sl_prime

        neg_local_emb = torch.cat(
            [neg_local_emb_l, neg_local_emb_l_prime], dim=-1)
        # indices_local = torch.empty(self.local_cand_neg,dtype = torch.long).random_(0,2*self.local_cand_neg)

        del neg_local_emb_l
        del neg_local_emb_l_prime

        neg_local_f_embed = torch.index_select(neg_local_emb, dim=-1, index=(
            torch.randperm(2*self.local_cand_neg)[:self.local_cand_neg]).to(device))

        # neg_local_f_embed -> (n_batch,self.n_pos_voxel,self.self.embedding_dim,self.local_cand_neg)
        # print(f"after calcuting local hard negatives")
        # gpu_usage()
        # step 9: sample global random samples from the batch
        # we have to sample n_global_neg samples for each positive voxel from the batch
        global_embedding_1st_ls_random = global_embedding_1st_ls.permute(
            1, 0, 2, 3, 4).reshape(self.embedding_dim, -1)
        global_embedding_2nd_ls_random = global_embedding_1st_ls.permute(
            1, 0, 2, 3, 4).reshape(self.embedding_dim, -1)
        # global_embedding_1st_ls_random.shape () -> (torch.Size([self.self.embedding_dim,
        # n_batch*self.n_pos_voxel*dim_z*dim_y*dim_x]))

        random_index = torch.empty(
            (n_batch*self.n_pos_voxel, self.global_n_random), dtype=torch.long, device=device)
        for idx in range(n_batch*self.n_pos_voxel):
            random_index[idx] = torch.randperm(global_embedding_2nd_ls_random.shape[1])[
                :self.global_n_random]

        # random_index = torch.empty((n_batch*self.n_pos_voxel,self.self.global_n_random),dtype = torch.long).random_(0,global_embedding_2nd_ls_random.shape[1])
        # (n_batch*self.n_pos_voxel,self.embedding_dim, self.global_n_random)
        random_index = random_index.unsqueeze(
            1).expand(-1, self.embedding_dim, -1)

        global_embedding_1st_ls_random = global_embedding_1st_ls_random.unsqueeze(
            0).expand(n_batch*self.n_pos_voxel, -1, -1)
        global_embedding_2nd_ls_random = global_embedding_2nd_ls_random.unsqueeze(
            0).expand(n_batch*self.n_pos_voxel, -1, -1)

        global_random_embed_l = torch.gather(
            global_embedding_1st_ls_random, -1, random_index).view(n_batch, self.n_pos_voxel, -1, self.global_n_random)
        global_random_embed_l_prime = torch.gather(
            global_embedding_2nd_ls_random, -1, random_index).view(n_batch, self.n_pos_voxel, -1, self.global_n_random)
        # global_random_embed_l.shape(n_batch*self.n_pos_voxel, self.self.self.embedding_dim,self.global_n_random)
        del global_embedding_2nd_ls_random
        del global_embedding_1st_ls_random
        global_random_embed = torch.cat(
            [global_random_embed_l, global_random_embed_l_prime], -1)

        # indices = torch.empty(self.global_n_random,dtype = torch.long).random_(0,2*self.global_n_random)

        global_random_embed = torch.index_select(global_random_embed, -1, index=(
            torch.randperm(2*self.global_n_random)[:self.global_n_random]).to(device))
        # global_random_embed -> torch.Size([n_batch, self.n_pos_voxel, self.self.embedding_dim, self.global_n_random])
        del global_random_embed_l_prime
        del global_random_embed_l

        # print(f"after calcuting Global random negatives")
        # gpu_usage()
        # 12. Loss Function
        # InfoNCE for global embedding
        total_sample = 1 + \
            global_random_embed.size(-1) + neg_glbal_f_embed.size(-1)
        grnd_global_neg_samples = torch.cat(
            [global_random_embed, neg_glbal_f_embed], dim=-1)

        # calculate similarity among positive voxel and corresponding negative voxels
        denom_global_loss = torch.einsum(
            'bij, bijk-> bik', [anchor_global_1st_ls, grnd_global_neg_samples])/self.los_temp
        # denom_global_loss.shap -> torch.Size([n_batch, self.n_pos_voxel, total_sample-1])

        # calculate similarity of anchor positive voxel with paired positive voxel

        nom_global_loss = (torch.einsum(
            'bij, bij-> bi', [anchor_global_1st_ls, anchor_global_2nd_ls])/self.los_temp).unsqueeze(-1)
        # nom_global_loss -> (n_batch*self.n_pos_voxel,1)

        # symmetric loss, compute samethings with f'
        denom_global_loss_fi = torch.einsum(
            'bij, bijk-> bik', [anchor_global_2nd_ls, grnd_global_neg_samples])/self.los_temp
        nom_global_loss_fi = (torch.einsum(
            'bij, bij-> bi', [anchor_global_2nd_ls, anchor_global_1st_ls])/self.los_temp).unsqueeze(-1)

        # deleting some variable
        del anchor_global_1st_ls
        del anchor_global_2nd_ls

        # add positive and negative voxels similarity
        denom_global_loss = torch.cat(
            [nom_global_loss, denom_global_loss], dim=-1)
        denom_global_loss_fi = torch.cat(
            [nom_global_loss_fi, denom_global_loss_fi], dim=-1)

        label_global = torch.zeros(denom_global_loss.shape[0], dtype=torch.long).to(
            device)  # (n_batch*self.self.n_pos_voxel)

        denom_global_loss = denom_global_loss.view(
            n_batch*self.n_pos_voxel, total_sample)
        denom_global_loss_fi = denom_global_loss_fi.view(
            n_batch*self.n_pos_voxel, total_sample)

        # denom_global_loss = self.global_softmax(denom_global_loss.view(n_batch*self.n_pos_voxel, total_sample))
        # print(f"denom_global_loss check :{denom_global_loss[0,:].sum()}")
        # InfoNCE for local embedding

        nom_local_loss = (torch.einsum(
            'bij, bij-> bi', [anchor_local_1st_ls, anchor_local_2nd_ls])/self.los_temp).unsqueeze(-1)

        denom_local_loss = torch.einsum(
            'bij, bijk-> bik', [anchor_local_1st_ls, neg_local_f_embed])/self.los_temp
        denom_local_loss = torch.cat(
            [nom_local_loss, denom_local_loss], dim=-1)

        # symmetric loss for f'
        nom_local_loss_fi = (torch.einsum(
            'bij, bij-> bi', [anchor_local_2nd_ls, anchor_local_1st_ls])/self.los_temp).unsqueeze(-1)

        denom_local_loss_fi = torch.einsum(
            'bij, bijk-> bik', [anchor_local_2nd_ls, neg_local_f_embed])/self.los_temp
        denom_local_loss_fi = torch.cat(
            [nom_local_loss_fi, denom_local_loss_fi], dim=-1)

        total_sample = denom_local_loss.size(-1)
        # print(f"denom_local_loss : {denom_local_loss.shape}")
        denom_local_loss = denom_local_loss.view(
            n_batch*self.n_pos_voxel, total_sample)
        denom_local_loss_fi = denom_local_loss_fi.view(
            n_batch*self.n_pos_voxel, total_sample)
        # denom_local_loss = self.local_softmax(denom_local_loss.view(n_batch*self.n_pos_voxel, total_sample))
        # print(f"denom_local_loss check :{denom_local_loss[0,:].sum()}")
        loss_label = torch.zeros(n_batch*self.n_pos_voxel, dtype=torch.long)

        # delete no more needed variables
        del anchor_local_1st_ls
        del anchor_local_2nd_ls

        del neg_local_f_embed
        del global_random_embed
        del neg_glbal_f_embed
        del grnd_global_neg_samples
        torch.cuda.empty_cache()

        return ([local_embedding_1st_ls, global_embedding_1st_ls],
                [local_embedding_2nd_ls, global_embedding_2nd_ls], [denom_local_loss, denom_local_loss_fi, denom_global_loss, denom_global_loss_fi, loss_label])


def get_model_for_inference(weight_path):

    n_batch = 1
    n_pos_voxel = 2
    global_n_neg = 2

    local_cand_neg = 2
    embedding_dim = 128
    global_n_random = 2
    los_temp = 0.5
    # we will use cpu for inference. As the image cannot be loaded into gpu.
    device = torch.device("cpu")
    radius = 3
    gpu_list = [torch.device("cuda:6"), torch.device("cuda:7")]
    gpu_list = [device, device]

    model = Model(n_batch, n_pos_voxel, global_n_neg, local_cand_neg,
                  embedding_dim, global_n_random, los_temp, device, radius=radius, gpu_list=gpu_list)

    lr = .0001
    weight_decay = 0.008
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    model.load_state_dict(torch.load(weight_path, map_location=device))
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    return model


def make_suitable_shape(shape):

    padding_amt = [32 - p % 32 if p % 32 != 0 else 0 for p in shape]

    padding_amt = [(p/2, p/2) if p % 2 == 0 else ((p+1)/2, p - (p+1)/2)
                   for p in padding_amt]
    padding_amt = [(int(p[0]), int(p[1])) for p in padding_amt]

    return padding_amt


def get_suitable_image(img_list):
    if isinstance(img_list, list):
        pre_pad_ls = []
        img_ls = []
        for img in img_list:
            padding_amt = make_suitable_shape(img.shape)
            pre = [p[0] for p in padding_amt]
            pre_pad_ls.append(pre)

            img = np.pad(img, padding_amt, mode='constant')
            img_ls.append(img)

        return pre_pad_ls, img_ls
    else:  # only single image
        padding_amt = make_suitable_shape(img_list)
        pre = [p[0] for p in padding_amt]
        img = np.pad(img_list, padding_amt)

        return [pre], [img]


'''
pre_pad,imgs = get_suitable_image([anchor_image, query_image])
achor_pre,query_pre= pre_pad
anchor,query = imgs
#anchor.shape ,query.shape
'''


def get_max_similar_coord(anchor_image, query_image, model, anchor_coordinates):
    '''
    based on image, it will give the coordinates of a point in query . This point is most similar to the achnor point in the anchor image.
    anchor_image, query_image are numpy image 

    Arguments:
    anchor_image: numpy image, (D,W,H)
    query_image: numpy image, (D,W,H)
    model : torch trained model, trained weight loaded
    anchor_coordinates: list. [x,y,z]

    Returns:
    tuple(tensor(x_cord),tensor(y_cord),tensor(z_cord))
    '''

    # get the padded image in suitable form
    pre_pad, imgs = get_suitable_image([anchor_image, query_image])
    achor_pre, query_pre = pre_pad
    anchor, query = imgs

    # preprocess images to have b, channel dimension and also have value in range of 0-1
    anchor = anchor/anchor.max()
    anchor = torch.from_numpy(anchor)
    anchor = anchor.unsqueeze(0).unsqueeze(0)
    anchor = anchor.to(torch.float)

    query = query/query.max()
    query = torch.from_numpy(query)
    query = query.unsqueeze(0).unsqueeze(0)
    query = query.to(torch.float)

    # get local, global embedding of anchor and query images from the trained model
    model.eval()
    local_embed, global_embed = model.fpn(anchor)
    local_embed, global_embed = F.normalize(
        local_embed, p=2.0, dim=1), F.normalize(global_embed, p=2.0, dim=1)

    local_embed_q, global_embed_q = model.fpn(query)
    local_embed_q, global_embed_q = F.normalize(
        local_embed_q, p=2.0, dim=1), F.normalize(global_embed_q, p=2.0, dim=1)

    # coordinates preprocessing
    post_anchor_embedding_cord = torch.tensor(
        anchor_coordinates).long() + torch.tensor(achor_pre).long()

    # adjust coordinates of anchors based on downsampling done in local and global embedding
    loc_cord = (post_anchor_embedding_cord//2).long()
    global_cord = post_anchor_embedding_cord.unsqueeze(
        -1)//torch.tensor([16, 16, 4]).unsqueeze(-1)
    global_cord = global_cord.long().squeeze()

    # get anchor embedding in local and global embedding
    anc_local_embed = local_embed[:, :, loc_cord[0], loc_cord[1], loc_cord[2]]
    anc_local_embed = anc_local_embed.unsqueeze(1)

    anc_global_embed = global_embed[:, :,
                                    global_cord[0], global_cord[1], global_cord[2]]
    anc_global_embed = anc_global_embed.unsqueeze(1)

    # compute similarity map of query image with anchor embedding
    sl = torch.einsum('bij, bjklm-> biklm', [anc_local_embed, local_embed_q])
    sg = torch.einsum('bij, bjklm-> biklm', [anc_global_embed, global_embed_q])

    # upsampling of sl,sg to query shape
    # upsampling
    D, H, W = query_image.shape
    up_sl = F.upsample(sl, size=(D, H, W), mode='trilinear')  # (b,c,d,w,h)
    up_sl = up_sl.squeeze().squeeze()  # (d,w,h)

    up_sg = F.upsample(sg, size=(D, H, W), mode='trilinear')  # (b,c,d,w,h)
    up_sg = up_sg.squeeze().squeeze()  # (d,w,h)

    # add up_sg,up_sl
    com = up_sl+up_sg  # (d,w,h)

    # now get the coordinates of the pick of the com
    highest_simi_cord = torch.where(com == torch.amax(com))

    extracted_simi_cord = [p.item() for p in highest_simi_cord]

    cord_native_space = torch.tensor(
        extracted_simi_cord).long() - torch.tensor(query_pre).long()
    return cord_native_space


def draw_similar_point(query_path_ls, point_ls):
    '''Draw highest similar point of query '''

    # first image is of template image
    query_images = [nib.load(query_path).get_fdata()
                    for query_path in query_path_ls]

    n_images = len(query_images)
    plt.figure()

    # subplot(r,c) provide the no. of rows and columns
    f, axarr = plt.subplots(n_images, 3)

    # use the created array to output your multiple images. In this case I have stacked 4 images vertically
    for i in range(n_images):
        img = query_images[i]
        pont = point_ls[i]

        axarr[i, 0].imshow(img[..., pont[2]])
        axarr[i, 0].plot(pont[0], pont[1], "o", color='red', zorder=2)

        axarr[i, 1].imshow(img[:, pont[1], :])
        axarr[i, 1].plot(pont[0], pont[2], "o", color='red', zorder=2)

        axarr[i, 2].imshow(img[pont[0], :, :], zorder=1)
        axarr[i, 2].plot(pont[1], pont[2], "o", color='red', zorder=2)
    f.subplots_adjust(left=0.125,
                      bottom=0.1,
                      right=0.9,
                      top=0.9,
                      wspace=0.2,
                      hspace=0.2)
    # f.tight_layout()
    plt.savefig('img.png')


def collate_fn(datas):
    '''
    data is a list [[patch_list,positive_pairs_ls],[patch_list,positive_pairs_ls]]
    return ([list_of_achors(1stpatch): n achors],[list_of_2nd_patch): n achors]], positive_pairs_ls_ls)
    '''
    data = [p[0] for p in datas]
    ids = [p[1] for p in datas]
    masks = [p[2] for p in datas]
    return torch.cat(data), torch.cat(ids), torch.cat(masks)


class patch_dataset_from_file_linear_MNI(Dataset):

    def __init__(self, imgs_paths, sample_dict, img_dir_dict):
        '''
        img_paths: list of folder_names.
        sample_dict: {int: folder_name}
        img_dir_dict: {'t1_unbiased_root_dit':, 
                        'converted_mask_root_dir:'}
        '''
        super(patch_dataset_from_file_linear_MNI, self).__init__()
        self.imgs_paths = imgs_paths
        self.sample_name_dict = {v: k for k, v in sample_dict.items()}
        self.img_dir_dict = img_dir_dict

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):

        img_path = self.imgs_paths[idx]
        sample_id = self.sample_name_dict[img_path]
        t1_unbiased_path = os.path.join(
            self.img_dir_dict['t1_unbiased_root_dit'], f"{img_path}", 'T1', 'T1_unbiased_brain_linear.nii.gz')

        img = nib.load(t1_unbiased_path).get_fdata()

        padding_amt = make_suitable_shape(img.shape)
        img = np.pad(img, padding_amt, mode='constant')

        # img = img/img.max()
        img = torch.from_numpy(img)
        max_intensity = torch.max(img)
        min_intensity = torch.min(img)
        mri_data_rescaled = (img - min_intensity) / \
            (max_intensity - min_intensity)

        img = mri_data_rescaled.unsqueeze(0).unsqueeze(0)
        img = img.to(torch.float)

        # masks
        mask_ls = []
        # this must be matched by sequence in which this two mask dict are save in extract_embed_linear_mni.py
        for mask in ['linear_T1_first_all_fast_firstseg.nii.gz', 'linear_T1_brain_seg.nii.gz']:
            mask_name = f"{img_path}_{mask}"

            mask_path = os.path.join(
                self.img_dir_dict['converted_mask_root_dir'], mask_name)
            # print(f"mask_path:{mask_path}")
            mask_ = nib.load(mask_path).get_fdata()
            mask_ = torch.from_numpy(mask_).unsqueeze(0)
            mask_ls.append(mask_)

        masks = torch.cat(mask_ls, dim=0)

        return img, torch.tensor([sample_id]), masks


class patch_dataset_from_file_Non_linear_MNI(Dataset):

    def __init__(self, imgs_paths, sample_dict, img_dir_dict):
        '''
        img_paths: list of folder_names.
        sample_dict: {int: folder_name}
        img_dir_dict: {'t1_unbiased_root_dit':, 
                        'converted_mask_root_dir:'}
        '''
        super(patch_dataset_from_file_Non_linear_MNI, self).__init__()
        self.imgs_paths = imgs_paths
        self.sample_name_dict = {v: k for k, v in sample_dict.items()}
        self.img_dir_dict = img_dir_dict

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):

        img_path = self.imgs_paths[idx]
        sample_id = self.sample_name_dict[img_path]
        t1_unbiased_path = os.path.join(
            self.img_dir_dict['t1_unbiased_root_dit'], f"{img_path}", 'T1', 'T1_brain_to_MNI.nii.gz')

        img = nib.load(t1_unbiased_path).get_fdata()

        padding_amt = make_suitable_shape(img.shape)
        img = np.pad(img, padding_amt, mode='constant')

        # img = img/img.max()
        img = torch.from_numpy(img)
        max_intensity = torch.max(img)
        min_intensity = torch.min(img)
        mri_data_rescaled = (img - min_intensity) / \
            (max_intensity - min_intensity)

        img = mri_data_rescaled.unsqueeze(0).unsqueeze(0)
        img = img.to(torch.float)

        # masks
        mask_ls = []
        # this must be matched by sequence in which this two mask dict are save in extract_embed_linear_mni.py
        for mask in ['T1_first_all_fast_firstseg_MNI.nii.gz', 'T1_brain_seg_MNI.nii.gz']:
            mask_name = f"{img_path}_{mask}"

            mask_path = os.path.join(
                self.img_dir_dict['converted_mask_root_dir'], mask_name)
            # print(f"mask_path:{mask_path}")
            mask_ = nib.load(mask_path).get_fdata()
            mask_ = torch.from_numpy(mask_).unsqueeze(0)
            mask_ls.append(mask_)

        masks = torch.cat(mask_ls, dim=0)

        return img, torch.tensor([sample_id]), masks


def save_embedding(save_dir, region_name, fsl_dict, patient_ids, embeddings, region_coords):
    '''
    save_dir: name of the directory in which all embeddings are saved
    region_name : one of the 'left_amygdala','right_amygdala', 'left_hippocampus','right_hippocampus'
    fsl_dict: dictionary of samples
    patient_ids: 1-d tensor of patients id in fsl_dict
    embeddings: tensor containing the embedding 
    region_coords: tuple(x_coords,y_coords,z_coords)


    '''
    region_dir = os.path.join(save_dir, region_name)

    for i, idx in enumerate(patient_ids):
        idx = idx.item()
        patient_id = fsl_dict[idx]
        temp_mni_full_name = patient_id.split("/")[-1]
        temp_mni_full_name = temp_mni_full_name.split("_")[:4]
        temp_mni_full_name = "_".join(temp_mni_full_name)
        pkl_file_name = os.path.join(
            region_dir, f"{temp_mni_full_name}_{region_name}.pkl")
        with open(pkl_file_name, 'wb') as f:
            pkl.dump([embeddings[i, ...], region_coords], f)
            print(
                f"done writing embedding for {region_name} region for {patient_id} sample in {pkl_file_name}")


def get_coords_one_region(region_code):
    '''Given the code in HarvardOxford parcel, give the voxel coordinates in MNI space'''

    warp_field_p = "<EXTERNAL: FSL HarvardOxford-sub-maxprob-thr50-1mm.nii.gz atlas>"
    harvad_parc = nib.load(warp_field_p).get_fdata()
    x, y, z = np.where(harvad_parc == region_code)

    return x, y, z


def get_hippocampus_amygdala_coords():

    region_code_dict = {'Left Hippocampus': 8+1,
                        'Right Hippocampus': 18+1,
                        'Right Amygdala': 19,
                        'Left Amygdala': 9
                        }
    lh_x, lh_y, lh_z = get_coords_one_region(
        region_code_dict['Left Hippocampus'])
    lh_x, lh_y, lh_z = [torch.from_numpy(p) for p in [lh_x, lh_y, lh_z]]

    rh_x, rh_y, rh_z = get_coords_one_region(
        region_code_dict['Right Hippocampus'])
    rh_x, rh_y, rh_z = [torch.from_numpy(p) for p in [rh_x, rh_y, rh_z]]

    ram_x, ram_y, ram_z = get_coords_one_region(
        region_code_dict['Right Amygdala'])
    ram_x, ram_y, ram_z = [torch.from_numpy(p) for p in [ram_x, ram_y, ram_z]]

    lam_x, lam_y, lam_z = get_coords_one_region(
        region_code_dict['Left Amygdala'])
    lam_x, lam_y, lam_z = [torch.from_numpy(p) for p in [lam_x, lam_y, lam_z]]

    return (lh_x, lh_y, lh_z), (rh_x, rh_y, rh_z), (ram_x, ram_y, ram_z), (lam_x, lam_y, lam_z)


def get_embedding(start_idx, end_idx):
    '''
    start_idx: any of the valid idx in fsl_dict:from 0 ... 10986-1
    end_idx : start_idx+2* any number.we are processing by batch of 3
    '''
    Ds, Ws, Hs = (182+10, 218+6, 182+10)

    # get hippocampus, amygdala coordinates
    saving_dir = cfg.embedding.bre_out_dir
    (lh_x_, lh_y_, lh_z_), (rh_x_, rh_y_, rh_z_), (ram_x_, ram_y_,
                                                   ram_z_), (lam_x_, lam_y_, lam_z_) = get_hippocampus_amygdala_coords()
    pre_x, pre_y, pre_z = torch.tensor([5]).long(), torch.tensor(
        [3]).long(), torch.tensor([5]).long()

    lh_x, lh_y, lh_z = lh_x_.long() + pre_x, lh_y_.long() + pre_y, lh_z_ + pre_z
    rh_x, rh_y, rh_z = rh_x_.long() + pre_x, rh_y_.long() + pre_y, rh_z_ + pre_z

    ram_x, ram_y, ram_z = ram_x_.long() + pre_x, ram_y_.long() + \
        pre_y, ram_z_ + pre_z
    lam_x, lam_y, lam_z = lam_x_.long() + pre_x, lam_y_.long() + \
        pre_y, lam_z_ + pre_z

    # get image dictionary

    pixel_root_path = "<EXTERNAL: project root for embedding extraction>"  # TODO

    img_path_ls = [fsl_dict[k] for k in range(start_idx, end_idx)]
    dataset = patch_dataset_from_file(img_path_ls, fsl_dict)

    ds = DataLoader(dataset, batch_size=batch_size,
                    shuffle=False, collate_fn=collate_fn, num_workers=3)

    # load model
    weight_path = os.path.join(cfg.embedding.weights_dir, "multi_state_dict_model.pt")
    model = get_model_for_inference(weight_path)
    device = torch.device('cpu')
    model.eval()

    for datas in ds:
        data, ids = datas
        print(ids)
        local_embed, global_embed = model.fpn(data)
        local_embed, global_embed = F.upsample(local_embed, size=(
            Ds, Ws, Hs), mode='trilinear'), F.upsample(global_embed, size=(Ds, Ws, Hs), mode='trilinear')
        local_embed, global_embed = F.normalize(
            local_embed, p=2.0, dim=1), F.normalize(global_embed, p=2.0, dim=1)
        local_embed, global_embed = local_embed.to(
            device), global_embed.to(device)

        com_embed = local_embed+global_embed

        # extract region embeddings
        left_hippocampus = com_embed[:, :, lh_x, lh_y, lh_z]
        right_hippocampus = com_embed[:, :, rh_x, rh_y, rh_z]

        right_amydala = com_embed[:, :, ram_x, ram_y, ram_z]
        left_amygdala = com_embed[:, :, lam_x, lam_y, lam_z]

        # save them

        region_name = "left_amygdala"
        save_embedding(saving_dir, region_name, fsl_dict, ids,
                       left_amygdala, (lam_x_, lam_y_, lam_z_))

        region_name = "right_amygdala"
        save_embedding(saving_dir, region_name, fsl_dict, ids,
                       right_amydala, (ram_x_, ram_y_, ram_z_))

        region_name = "left_hippocampus"
        save_embedding(saving_dir, region_name, fsl_dict, ids,
                       left_hippocampus, (lh_x_, lh_y_, lh_z_))

        region_name = "right_hippocampus"
        save_embedding(saving_dir, region_name, fsl_dict, ids,
                       right_hippocampus, (rh_x_, rh_y_, rh_z_))


def get_filename(full_path):
    return full_path.split("/")[-1].split(".")[0]


def compute_region_embedding(embedding, mask, region_id, sigma=1.5):
    """
    Computes region embedding for a given region in the mask.

    Args:
        embedding: Tensor of shape (B, embed_dim, W, H, D), voxel embeddings.
        mask: Tensor of shape (B, W, H, D), subcortical mask with region labels.
        region_id: Integer, ID of the region to compute embedding for.
        sigma: Float, controls Gaussian weight decay for border voxels.

    Returns:
        region_embedding: Tensor of shape (B, embed_dim), the region embedding.
    """

    if not isinstance(embedding, torch.Tensor):
        embedding = torch.from_numpy(embedding)

    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(mask).to(embedding.device)

    if len(embedding.shape) != 5:
        embedding = embedding.unsqueeze(0)

    if len(mask.shape) != 4:  # meanig only one sample is given
        mask = mask.unsqueeze(0)

    B, embed_dim, W, H, D = embedding.shape

    # Binary mask for the region
    region_mask = (mask == region_id).float()  # Shape: (B, W, H, D)

    # Compute distance to the nearest border (outside the region) # Shape: (B, W, H, D)
    distance = torch.tensor(distance_transform_edt(
        region_mask.cpu().numpy()), device=embedding.device)
    weights = torch.exp(-distance**2 / (2*sigma**2))
    region_mask = region_mask.to(embedding.device)
    # apply weights to embeddings
    region_mask = region_mask.unsqueeze(1)  # shape: (B,1, W,H,D)
    weights = weights.unsqueeze(1)
    weighted_embedding = embedding*region_mask*weights

    region_embedding = weighted_embedding.sum(
        dim=(2, 3, 4))/(weights * region_mask).sum(dim=(2, 3, 4))

    return region_embedding.squeeze()


def extract_region_from_matrix(idx, sample_dict, region_template, region_template_dict, mri_matrix, save_dir):

    # folder_name
    folder_name = sample_dict[idx]
    folder_path = os.path.join(save_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # print(f"folder_name : {folder_name}")

    # iterate over each region of each dictionary

    for region_id in region_template_dict.keys():
        # extract x,y,z of the region coordinates.

        x, y, z = np.where(region_template == region_id)

        # region_embedding_mean = compute_region_embedding(mri_matrix, region_template, region_id, sigma= 1.5).cpu()

        region_embedding_mean = mri_matrix[:, x, y, z].mean(dim=-1).cpu()

        '''
        #now save them
        sum_path = f"{folder_name}_{region_template_dict[region_id]}_sum.pt"
        sum_path = os.path.join(folder_path,sum_path)
        torch.save(region_embedding_sum,sum_path)
        '''

        # now save them
        mean_path = f"{folder_name}_{region_id}_{region_template_dict[region_id]}_mean.pt"
        mean_path = os.path.join(folder_path, mean_path)
        torch.save(region_embedding_mean, mean_path)
    print()


def generate_embed_from_model(model, data, original_shape, embedding_type='combined', do_upsampling=True):
    ''' Generate the embedding using the trained model.
        Constraints:
        model must be the FPN module.
        data and model should be in the same device
        Arguments:
            original_shape: (W,H,D). original shape of the MRI.
    '''
    # get FPN output

    with torch.no_grad():

        local_embed, global_embed = model(data)

        # interpolate to original shape
        if do_upsampling:
            local_embed, global_embed = F.upsample(local_embed, size=tuple(original_shape), mode='trilinear'), F.upsample(
                global_embed, size=tuple(original_shape), mode='trilinear')

        # get the embedding type
        if embedding_type == 'combined':
            print(f"Embedding type: combined")
            com_embed = local_embed+global_embed

        elif embedding_type == 'local':
            print(f"Embedding type: local")
            com_embed = local_embed

        else:
            print(f"Embedding type: global_embed")
            com_embed = global_embed
        return com_embed


def modify_folder_name(prev_folder_name, new_suffix):

    basename = os.path.basename(prev_folder_name)
    dirname = os.path.dirname(prev_folder_name)
    suffix = basename.split("_")[-1]
    basename = basename.replace(suffix, new_suffix)

    new_name = os.path.join(dirname, basename)
    os.makedirs(new_name, exist_ok=True)
    return new_name


def get_embedding_all_region(start_idx, end_idx, saving_dir,
                             pixel_root_path, weight_path, region_dict_dicts_path, img_dir_dict,
                             sample_dict_name='mni_img.pkl', device='cpu', batch_size=5,
                             embedding_type='combined', random_weight=False
                             ):
    '''
    start_idx: int, sample_dict_names keys starts from 0. start_idx is the index from which we want to start
    end_idx : int, start_idx is the index from which we want to start.
    saving_dir = path string. the directory in which we will save our extracted region embedding. extracted 
                 image will be saved on a subfolder named by patient.
    pixel_root_path: path string. it is the root directory path of the project from where all the relative paths
                     are derived.
    weight_path: .pt file path of model weights.
    region_dict_dicts: path of dict of dictionary. {0: brain_seg(w,g,csf),1: first segmentations}
    img_dir_dict: dict. {'linear_unbiased_mri_dir':,'converted_mask_dir'}
    sample_dict_name: dict.{0: patient1_uique_id, 1: patient2_uique_id}
    '''

    Ds, Ws, Hs = (182+10, 218+6, 182+10)
    print(f"{device}=>{sample_dict_name}")

    # get hippocampus, amygdala coordinates
    saving_dir = saving_dir

    # get dict of dictionary of regions.

    if not os.path.exists(region_dict_dicts_path):
        print(f"No region_dict_dicts  exists, creating....")
        save_region_dict(region_dict_dicts_path)

    with open(region_dict_dicts_path, 'rb') as f:
        region_dict_dicts = pkl.load(f)

    pixel_root_path = pixel_root_path
    # fsl_dict = None  #TODO make a 0:image_path/image_name dict

    # invalide_sample_list = list(range(0,4546))
    # invalide_sample_list.append(30630)

    # invalide_sample_list = [4545,30630]
    invalide_sample_list = []
    with open(sample_dict_name, 'rb') as f:
        fsl_dict = pkl.load(f)

    # currently original_fsl dict is {int:full_path_of_unbiased_linear_MNI}.
    # but this method expect only the {int:patient_unique id}

    fsl_dict = {k: v.split("/")[-3] for k, v in fsl_dict.items()}

    # exclude file ids for which first segmentation mask is not done properly
    fsl_temp = {v: k for k, v in fsl_dict.items()}
    # subject IDs to exclude (failed FIRST segmentation); supply via config
    excluded_ids = []  # load QC exclusion list from config
    for excluding_id in excluded_ids:
        if excluding_id in fsl_temp:
            invalide_sample_list.append(fsl_temp[excluding_id])
            del fsl_temp[excluding_id]

    fsl_dict = {v: k for k, v in fsl_temp.items()}

   # (QC exclusion list is loaded from config, not hard-coded)
    # del fsl_dict[7839] # delete this if, start_idx is 0

    img_path_ls = [fsl_dict[k] for k in range(
        start_idx, end_idx) if k not in invalide_sample_list]

    # valid-sample list output; set path via config
    with open("<EXTERNAL: valid sample list output path>", 'wb') as f:
        pkl.dump(img_path_ls, f)

    if 'MASKS_MNI_NON_LINEAR' in img_dir_dict['converted_mask_root_dir']:
        print(f"creatig dataset from non-linear masks")
        dataset = patch_dataset_from_file_Non_linear_MNI(imgs_paths=img_path_ls, sample_dict=fsl_dict,
                                                         img_dir_dict=img_dir_dict)

    elif 'MASKS_MNI_LINEAR' in img_dir_dict['converted_mask_root_dir']:
        print(f"creatig dataset from linear masks")
        dataset = patch_dataset_from_file_linear_MNI(imgs_paths=img_path_ls, sample_dict=fsl_dict,
                                                     img_dir_dict=img_dir_dict)

    ds = DataLoader(dataset, batch_size=batch_size,
                    shuffle=False, collate_fn=collate_fn, num_workers=5)

    # load model
    weight_path = weight_path  # TODO, set checkpoint path via config

    if random_weight is not True:
        print("Trained model is being loaded")
        model = models.get_pretrained_model_voxel_classification_model(
            weight_path, full_wt_path_given=True)
    else:
        print("Random weight is loaded")
        model = models.get_pretrained_model_voxel_classification_model(weight_path, load_pretrained_wt=False,
                                                                       full_wt_path_given=False, fully_random_weight=True)

    device = torch.device(device)
    model.to(device)
    model.eval()

    x_lb = 5
    x_rb = 5
    y_lb = 3
    y_rb = 3
    z_lb = 5
    z_rb = 5
    # print(model)
    with torch.no_grad():
        for datas in ds:
            data, ids, masks = datas

            data = data.to(device)
            # masks = masks.to(device)
            # print(f"{ids.shape} => {ids}")
            # print(f"data shape: {data.shape} ")

            local_embed, global_embed = model(data)

            local_embed, global_embed = F.upsample(local_embed, size=(
                Ds, Ws, Hs), mode='trilinear'), F.upsample(global_embed, size=(Ds, Ws, Hs), mode='trilinear')
            # local_embed,global_embed = F.normalize(local_embed,p=2.0,dim= 1),F.normalize(global_embed,p=2.0,dim=1)
            # print(f"local_embed: {local_embed.shape}, global_embed :{global_embed.shape} ")
            # local_embed,global_embed = local_embed.to(device),global_embed.to(device)
            '''
            print(f"after computing normalization of local and global embedding")             
            gpu_usage() 
            print()
            print()             
            '''
            for embedding_type in ['combined', 'local', 'global']:
                saving_dir = modify_folder_name(saving_dir, embedding_type)
                if embedding_type == 'combined':
                    print(f"Embedding type: combined")
                    com_embed = local_embed+global_embed

                elif embedding_type == 'local':
                    print(f"Embedding type: local")
                    com_embed = local_embed

                else:
                    print(f"Embedding type: global_embed")
                    com_embed = global_embed

                # com_embed = com_embed[:,:,5:-5,3:-3,5:-5]
                com_embed = com_embed[:, :, x_lb:-x_rb, y_lb:-y_rb, z_lb:-z_rb]
                # com_embed = com_embed.cpu()
                # remove the padding
                print(f"com_embed: {com_embed.shape}")

                # now saving

                for idx in range(len(com_embed)):
                    file_id = ids[idx].item()
                    '''
                    file_name = f"{get_filename(fsl_dict[file_id])}.pt"
                    save_path  = os.path.join(saving_dir,file_name) 
                    torch.save(com_embed[idx],save_path)
                    '''

                    for region_dict_id in region_dict_dicts.keys():
                        # subcortical or gm,wm,csf
                        region_template = masks[region_dict_id]

                        region_id_label_dict = region_dict_dicts[region_dict_id]
                        print(f"{region_dict_id}=> {region_id_label_dict}")
                        extract_region_from_matrix(idx=file_id, sample_dict=fsl_dict,
                                                   region_template=region_template,
                                                   region_template_dict=region_id_label_dict,
                                                   mri_matrix=com_embed[idx],
                                                   save_dir=saving_dir)
            print(f"Done extracting region embedding from {file_id}")
            # print(f"done extracing {file_id}")
            # save_embedding(saving_dir, region_name, fsl_dict,ids, left_amygdala,(lam_x_,lam_y_,lam_z_))
# --------


def get_embedding_all_region_pixpro(start_idx, end_idx, saving_dir,
                                    pixel_root_path, weight_path, region_dict_dicts_path, img_dir_dict,
                                    sample_dict_name='mni_img.pkl', device='cpu', batch_size=5,
                                    embedding_type='combined', random_weight=False, config_file=""
                                    ):
    '''
    start_idx: int, sample_dict_names keys starts from 0. start_idx is the index from which we want to start
    end_idx : int, start_idx is the index from which we want to start.
    saving_dir = path string. the directory in which we will save our extracted region embedding. extracted 
                 image will be saved on a subfolder named by patient.
    pixel_root_path: path string. it is the root directory path of the project from where all the relative paths
                     are derived.
    weight_path: .pt file path of model weights.
    region_dict_dicts: path of dict of dictionary. {0: brain_seg(w,g,csf),1: first segmentations}
    img_dir_dict: dict. {'linear_unbiased_mri_dir':,'converted_mask_dir'}
    sample_dict_name: dict.{0: patient1_uique_id, 1: patient2_uique_id}
    '''

    Ds, Ws, Hs = (182+10, 218+6, 182+10)
    print(f"{device}=>{sample_dict_name}")

    # get hippocampus, amygdala coordinates
    saving_dir = saving_dir

    # get dict of dictionary of regions.

    if not os.path.exists(region_dict_dicts_path):
        print(f"No region_dict_dicts  exists, creating....")
        save_region_dict(region_dict_dicts_path)

    with open(region_dict_dicts_path, 'rb') as f:
        region_dict_dicts = pkl.load(f)

    pixel_root_path = pixel_root_path
    # fsl_dict = None  #TODO make a 0:image_path/image_name dict

    # invalide_sample_list = list(range(0,4546))
    # invalide_sample_list.append(30630)

    # invalide_sample_list = [4545,30630]
    invalide_sample_list = []
    with open(sample_dict_name, 'rb') as f:
        fsl_dict = pkl.load(f)

    # currently original_fsl dict is {int:full_path_of_unbiased_linear_MNI}.
    # but this method expect only the {int:patient_unique id}

    fsl_dict = {k: v.split("/")[-3] for k, v in fsl_dict.items()}

    # exclude file ids for which first segmentation mask is not done properly
    fsl_temp = {v: k for k, v in fsl_dict.items()}
    # subject IDs to exclude (failed FIRST segmentation); supply via config
    excluded_ids = []  # load QC exclusion list from config
    for excluding_id in excluded_ids:
        if excluding_id in fsl_temp:
            invalide_sample_list.append(fsl_temp[excluding_id])
            del fsl_temp[excluding_id]

    fsl_dict = {v: k for k, v in fsl_temp.items()}

   # (QC exclusion list is loaded from config, not hard-coded)
    # del fsl_dict[7839] # delete this if, start_idx is 0

    img_path_ls = [fsl_dict[k] for k in range(
        start_idx, end_idx) if k not in invalide_sample_list]

    if 'MASKS_MNI_NON_LINEAR' in img_dir_dict['converted_mask_root_dir']:
        print(f"creatig dataset from non-linear masks")
        dataset = patch_dataset_from_file_Non_linear_MNI(imgs_paths=img_path_ls, sample_dict=fsl_dict,
                                                         img_dir_dict=img_dir_dict)

    elif 'MASKS_MNI_LINEAR' in img_dir_dict['converted_mask_root_dir']:
        print(f"creatig dataset from linear masks")
        dataset = patch_dataset_from_file_linear_MNI(imgs_paths=img_path_ls, sample_dict=fsl_dict,
                                                     img_dir_dict=img_dir_dict)

    ds = DataLoader(dataset, batch_size=batch_size,
                    shuffle=False, collate_fn=collate_fn, num_workers=5)

    # load model
    weight_path = weight_path  # TODO, set checkpoint path via config

    if random_weight is not True:
        print("Trained model is being loaded")
        experiment_config_path = os.path.join(
            config.pixpro_config_folder, config_file)
        experiment_config = config_loader_pixpro.load_config(
            experiment_config_path)
        model = pixpro2.get_model(
            experiment_config, do_inference=True, weight_path=weight_path, device=device)

        # pixpro.get_pretrained_model_voxel_classification_model(weight_path,full_wt_path_given=True)

    else:
        raise ValueError('Random weight is not still implemented')
        print("Random weight is loaded")
        model = models.get_pretrained_model_voxel_classification_model(weight_path, load_pretrained_wt=False,
                                                                       full_wt_path_given=False, fully_random_weight=True)

    device = torch.device(device)
    model.to(device)
    model.eval()

    x_lb = 5
    x_rb = 5
    y_lb = 3
    y_rb = 3
    z_lb = 5
    z_rb = 5
    # print(model)
    with torch.no_grad():
        for datas in ds:
            data, ids, masks = datas

            data = data.to(device)
            # masks = masks.to(device)
            # print(f"{ids.shape} => {ids}")
            # print(f"data shape: {data.shape} ")

            _, com_embed, _ = model(data)

            # , global_embed = F.upsample(local_embed, size=(Ds,Ws,Hs), mode='trilinear'), F.upsample(global_embed, size=(Ds,Ws,Hs), mode='trilinear')
            # local_embed,global_embed = F.normalize(local_embed,p=2.0,dim= 1),F.normalize(global_embed,p=2.0,dim=1)
            # print(f"local_embed: {local_embed.shape}, global_embed :{global_embed.shape} ")
            # local_embed,global_embed = local_embed.to(device),global_embed.to(device)
            '''
            print(f"after computing normalization of local and global embedding")             
            gpu_usage() 
            print()
            print()             
            '''

            # com_embed = com_embed[:,:,5:-5,3:-3,5:-5]
            com_embed = com_embed[:, :, x_lb:-x_rb, y_lb:-y_rb, z_lb:-z_rb]
            # com_embed = com_embed.cpu()
            # remove the padding
            print(f"com_embed: {com_embed.shape}")

            # now saving

            for idx in range(len(com_embed)):
                file_id = ids[idx].item()
                '''
                file_name = f"{get_filename(fsl_dict[file_id])}.pt"
                save_path  = os.path.join(saving_dir,file_name) 
                torch.save(com_embed[idx],save_path)
                '''

                for region_dict_id in region_dict_dicts.keys():
                    # subcortical or gm,wm,csf
                    region_template = masks[region_dict_id]

                    region_id_label_dict = region_dict_dicts[region_dict_id]
                    print(f"{region_dict_id}=> {region_id_label_dict}")
                    extract_region_from_matrix(idx=file_id, sample_dict=fsl_dict,
                                               region_template=region_template,
                                               region_template_dict=region_id_label_dict,
                                               mri_matrix=com_embed[idx],
                                               save_dir=saving_dir)
            print(f"Done extracting region embedding from {file_id}")
            # print(f"done extracing {file_id}")
            # save_embedding(saving_dir, region_name, fsl_dict,ids, left_amygdala,(lam_x_,lam_y_,lam_z_))


# ----------


def wrapper_get_embedding_all_region(start_index, end_index,
                                     grand_root_path, region_dict_dicts_path,
                                     img_dir_dict, sample_dict_name='mni_img.pkl', device='cpu', types='',
                                     gwas_folder_name=None,
                                     check_point_name='test_experiment_model_3_new_1_model_3_96_96_96_s_-1_p_50_llf0_loss_7.455170304167504.pt',

                                     embedding_type='combined', random_weight=False, **kwargs):
    '''
    start_idx, end_idx, saving_dir, 
                                pixel_root_path, weight_path,region_dict_dicts_path, img_dir_dict,
                                sample_dict_name = 'mni_img.pkl',device = 'cpu', batch_size = 5

    '''

    grand_root_path = grand_root_path
    if gwas_folder_name is None:
        embed_save_dir = os.path.join(
            config.data_path(), 'RBGWAS_RELATED_DATA')
    else:
        if random_weight is not True:
            embed_save_dir = os.path.join(config.data_path(
            ), 'RBGWAS_RELATED_DATA', gwas_folder_name, kwargs['sub_folder'])
            os.makedirs(embed_save_dir, exist_ok=True)
        else:
            embed_save_dir = os.path.join(config.data_path(
            ), 'RBGWAS_RELATED_DATA', f"{gwas_folder_name}_random", kwargs['sub_folder'])
            os.makedirs(embed_save_dir, exist_ok=True)

    if len(types) == 0:
        saving_dir = os.path.join(embed_save_dir, 'LINEAR_MNI')
        if os.path.exists(saving_dir):
            shutil.rmtree(saving_dir)
        os.makedirs(saving_dir)
        print(f'Embedding will be saved in {saving_dir}')

    else:

        if 'MASKS_MNI_NON_LINEAR' in img_dir_dict['converted_mask_root_dir']:
            saving_dir = os.path.join(
                embed_save_dir, f'NON_LINEAR_MNI_{types}_{embedding_type}')

        elif 'MASKS_MNI_LINEAR' in img_dir_dict['converted_mask_root_dir']:
            saving_dir = os.path.join(
                embed_save_dir, f'LINEAR_MNI_{types}_{embedding_type}')

        if kwargs['delete_folder'] and os.path.exists(saving_dir):
            if os.path.exists(saving_dir):
                print(f"{saving_dir} exists. delelting first...")
                shutil.rmtree(saving_dir)
        elif not kwargs['delete_folder'] and not os.path.exists(saving_dir):
            print(f"{saving_dir} does not exists. Creating...")
            os.makedirs(saving_dir)
        print(f'Embedding will be saved in {saving_dir}')

    # os.makedirs(saving_dir,exist_ok=True)

    start_idx = start_index
    end_idx = end_index

    pixel_root_path = grand_root_path

    if gwas_folder_name is None:
        weight_path_dir = os.path.join(grand_root_path, 'model_weights')
    else:
        weight_path_dir = os.path.join(
            grand_root_path, 'model_weights', gwas_folder_name)

    # os.makedirs(weight_path_dir,exist_ok=True)
    weight_path = os.path.join(weight_path_dir, check_point_name)

    device = device
    batch_size = 1
    print(
        f"started extracing MRI embeddings of {start_idx}:{end_idx} of {sample_dict_name}")

    get_embedding_all_region(start_idx, end_idx, saving_dir,
                             pixel_root_path, weight_path, region_dict_dicts_path,
                             img_dir_dict, sample_dict_name=sample_dict_name,
                             device=device, batch_size=batch_size,
                             embedding_type=embedding_type, random_weight=random_weight)
# ------


def wrapper_get_embedding_all_region_pixpro(start_index, end_index,
                                            grand_root_path, region_dict_dicts_path,
                                            img_dir_dict, sample_dict_name='mni_img.pkl', device='cpu', types='',
                                            gwas_folder_name=None,
                                            check_point_name='test_experiment_model_3_new_1_model_3_96_96_96_s_-1_p_50_llf0_loss_7.455170304167504.pt',

                                            embedding_type='combined', random_weight=False,
                                            config_file="",
                                            **kwargs):
    '''
    start_idx, end_idx, saving_dir, 
                                pixel_root_path, weight_path,region_dict_dicts_path, img_dir_dict,
                                sample_dict_name = 'mni_img.pkl',device = 'cpu', batch_size = 5

    '''

    grand_root_path = config.project_dir
    if gwas_folder_name is None:
        embed_save_dir = os.path.join(
            config.data_path, 'RBGWAS_RELATED_DATA')
    else:
        if random_weight is not True:
            embed_save_dir = os.path.join(
                config.data_path, 'RBGWAS_RELATED_DATA', gwas_folder_name, kwargs['sub_folder'])
            os.makedirs(embed_save_dir, exist_ok=True)
        else:
            embed_save_dir = os.path.join(
                config.data_path, 'RBGWAS_RELATED_DATA', f"{gwas_folder_name}_random", kwargs['sub_folder'])
            os.makedirs(embed_save_dir, exist_ok=True)

    if len(types) == 0:
        saving_dir = os.path.join(embed_save_dir, 'LINEAR_MNI')
        if os.path.exists(saving_dir):
            shutil.rmtree(saving_dir)
        os.makedirs(saving_dir)
        print(f'Embedding will be saved in {saving_dir}')

    else:

        if 'MASKS_MNI_NON_LINEAR' in img_dir_dict['converted_mask_root_dir']:
            saving_dir = os.path.join(
                embed_save_dir, f'NON_LINEAR_MNI_{types}_{embedding_type}')

        elif 'MASKS_MNI_LINEAR' in img_dir_dict['converted_mask_root_dir']:
            saving_dir = os.path.join(
                embed_save_dir, f'LINEAR_MNI_{types}_{embedding_type}')

        if kwargs['delete_folder'] and os.path.exists(saving_dir):
            if os.path.exists(saving_dir):
                print(f"{saving_dir} exists.  first...")
                #shutil.rmtree(saving_dir)
        elif not kwargs['delete_folder'] and not os.path.exists(saving_dir):
            print(f"{saving_dir} does not exists. Creating...")
            os.makedirs(saving_dir)
        print(f'Embedding will be saved in {saving_dir}')

    # os.makedirs(saving_dir,exist_ok=True)

    start_idx = start_index
    end_idx = end_index

    pixel_root_path = grand_root_path

    if gwas_folder_name is None:
        weight_path_dir = os.path.join(grand_root_path, 'model_weights_pixpro')
    else:
        weight_path_dir = os.path.join(
            grand_root_path, 'model_weights_pixpro', kwargs['sub_folder'])

    # os.makedirs(weight_path_dir,exist_ok=True)
    weight_path = os.path.join(weight_path_dir, check_point_name)

    device = device
    batch_size = 1
    print(
        f"started extracing MRI embeddings of {start_idx}:{end_idx} of {sample_dict_name}")

    get_embedding_all_region_pixpro(start_idx, end_idx, saving_dir,
                                    pixel_root_path, weight_path, region_dict_dicts_path,
                                    img_dir_dict, sample_dict_name=sample_dict_name,
                                    device=device, batch_size=batch_size,
                                    embedding_type=embedding_type, random_weight=random_weight,
                                    config_file=config_file)


# ---------
def driver_wrapper_get_embedding_all_region(start_idx=0, end_index=-1, types='discovery',
                                            device=torch.device('cuda:1'),
                                            gwas_folder_name=None,
                                            check_point_name='test_experiment_model_3_new_1_model_3_96_96_96_s_-1_p_50_llf0_loss_7.455170304167504.pt',
                                            embedding_type='combined', random_weight=False,
                                            **kwargs):

    grand_root_path = config.root_path()
    start_idx = start_idx
    # end_index = 10000
    region_dict_dicts_path = os.path.join(config.data_path(
    ), 'RBGWAS_RELATED_DATA', 'region_dict_lin_min.pkl')  # 'region_dict_lin_min.pkl'
    if 'mni_type' in kwargs:
        if kwargs['mni_type'] == 'linear':
            print(f"We are using linear masks")
            converted_mask_root_dir = os.path.join(
                config.data_path(), 'RBGWAS_RELATED_DATA', 'MASKS_MNI_LINEAR')
        elif kwargs['mni_type'] == 'non-linear':
            print(f"We are using Non-linear masks")
            converted_mask_root_dir = os.path.join(
                config.data_path(), 'RBGWAS_RELATED_DATA', 'MASKS_MNI_NON_LINEAR')

    else:  # just for legacy code compatibility
        converted_mask_root_dir = os.path.join(
            config.data_path(), 'RBGWAS_RELATED_DATA', 'MASKS_MNI_LINEAR')
    # os.makedirs(converted_mask_root_dir,exist_ok=True)

    img_dir_dict = {'t1_unbiased_root_dit': cfg.embedding.t1_dir,
                    'converted_mask_root_dir': converted_mask_root_dir}

    print(f"mask path: {img_dir_dict['converted_mask_root_dir']}")

    if types != "remaining":
        sample_dict_name = 'discovery_gwas_sample_dict.pkl' if types == 'discovery' else 'replication_gwas_sample_dict.pkl'
    else:
        sample_dict_name = 'remaining_gwas_sample_dict.pkl'

    # the sample dict is created by inference_cuda_3.create_sample_dict(). first create it .

    if gwas_folder_name is None:
        sample_dict_name = os.path.join(config.data_path(), sample_dict_name)
    else:
        if random_weight is not True:
            f_path = os.path.join(config.data_path(), gwas_folder_name)
            os.makedirs(f_path, exist_ok=True)
            sample_dict_name = os.path.join(f_path, sample_dict_name)
        else:
            f_path = os.path.join(config.data_path(),
                                  f"{gwas_folder_name}_random")
            os.makedirs(f_path, exist_ok=True)
            sample_dict_name = os.path.join(f_path, sample_dict_name)
    if end_index == -1:
        with open(sample_dict_name, 'rb') as f:
            end_index = len(pkl.load(f))

    print(f"checkpoint path :{check_point_name}")
    # device = 'cuda:1'

    print(f"Extracting embedding for {types}")
    print(f"sample dictionary path is :{sample_dict_name}")
    print(
        f"Extracting embedding from {start_idx}th sample to {end_index}th sample")

    wrapper_get_embedding_all_region(start_index=start_idx, end_index=end_index,
                                     grand_root_path=grand_root_path, region_dict_dicts_path=region_dict_dicts_path,
                                     img_dir_dict=img_dir_dict, sample_dict_name=sample_dict_name, device=device,
                                     types=types, gwas_folder_name=gwas_folder_name, check_point_name=check_point_name,
                                     embedding_type=embedding_type, random_weight=random_weight,
                                     sub_folder=kwargs['sub_folder'],
                                     delete_folder=kwargs['delete_folder'],

                                     )


# ---
def driver_wrapper_get_embedding_all_region_pixpro(start_idx=0, end_index=-1, types='discovery',
                                                   device=torch.device(
                                                       'cuda:1'),
                                                   gwas_folder_name=None,
                                                   check_point_name='test_experiment_model_3_new_1_model_3_96_96_96_s_-1_p_50_llf0_loss_7.455170304167504.pt',
                                                   embedding_type='combined', random_weight=False,
                                                   config_file="",
                                                   **kwargs):

    grand_root_path = config.project_dir
    start_idx = start_idx
    # end_index = 10000
    region_dict_dicts_path = os.path.join(config.data_path,
                                          'RBGWAS_RELATED_DATA', 'region_dict_lin_min.pkl')  # 'region_dict_lin_min.pkl'
    if 'mni_type' in kwargs:
        if kwargs['mni_type'] == 'linear':
            print(f"We are using linear masks")
            converted_mask_root_dir = os.path.join(
                config.data_path, 'RBGWAS_RELATED_DATA', 'MASKS_MNI_LINEAR')
        elif kwargs['mni_type'] == 'non-linear':
            print(f"We are using Non-linear masks")
            converted_mask_root_dir = os.path.join(
                config.data_path, 'RBGWAS_RELATED_DATA', 'MASKS_MNI_NON_LINEAR')

    else:  # just for legacy code compatibility
        converted_mask_root_dir = os.path.join(
            config.data_path, 'RBGWAS_RELATED_DATA', 'MASKS_MNI_LINEAR')
    # os.makedirs(converted_mask_root_dir,exist_ok=True)

    img_dir_dict = {'t1_unbiased_root_dit': cfg.embedding.t1_dir,
                    'converted_mask_root_dir': converted_mask_root_dir}

    print(f"mask path: {img_dir_dict['converted_mask_root_dir']}")

    if types != "remaining":
        sample_dict_name = 'discovery_gwas_sample_dict.pkl' if types == 'discovery' else 'replication_gwas_sample_dict.pkl'
    else:
        sample_dict_name = 'remaining_gwas_sample_dict.pkl'

    # the sample dict is created by inference_cuda_3.create_sample_dict(). first create it .

    if gwas_folder_name is None:
        sample_dict_name = os.path.join(config.data_path, sample_dict_name)
    else:
        if random_weight is not True:
            f_path = os.path.join(config.data_path, gwas_folder_name)
            os.makedirs(f_path, exist_ok=True)
            sample_dict_name = os.path.join(f_path, sample_dict_name)

        else:
            f_path = os.path.join(config.data_path,
                                  f"{gwas_folder_name}_random")
            os.makedirs(f_path, exist_ok=True)
            sample_dict_name = os.path.join(f_path, sample_dict_name)
    if end_index == -1:
        with open(sample_dict_name, 'rb') as f:
            end_index = len(pkl.load(f))

    print(f"checkpoint path :{check_point_name}")
    # device = 'cuda:1'

    print(f"Extracting embedding for {types}")
    print(f"sample dictionary path is :{sample_dict_name}")
    print(
        f"Extracting embedding from {start_idx}th sample to {end_index}th sample")

    wrapper_get_embedding_all_region_pixpro(start_index=start_idx, end_index=end_index,
                                            grand_root_path=grand_root_path, region_dict_dicts_path=region_dict_dicts_path,
                                            img_dir_dict=img_dir_dict, sample_dict_name=sample_dict_name, device=device,
                                            types=types, gwas_folder_name=gwas_folder_name, check_point_name=check_point_name,
                                            embedding_type=embedding_type, random_weight=random_weight,
                                            sub_folder=kwargs['sub_folder'],
                                            delete_folder=kwargs['delete_folder'],
                                            config_file=config_file
                                            )


# ---


# extracting MNI space MRIs from ziqian folder
def extract_t1_brain(save_dir, file_type="T1/T1_brain_to_MNI.nii.gz", ukb_dir="<EXTERNAL: UKB T1/T2 zip source dir>"):
    '''
    Extract number_t1 of T1 brain from the UKB imaging release to save_dir. Individual
    folder is identified patientid_20252


    '''
    files = [p for p in os.listdir(ukb_dir) if '_20252_2_0.zip' in p]

    print(f"we are extracting {file_type} for total {len(files)} samples")
    for file in files:

        with ZipFile(os.path.join(ukb_dir, file)) as zipobj:

            listOfFileNames = zipobj.namelist()

            for fs in listOfFileNames:
                if fs in [file_type]:

                    save_path = file.split(".")[0]

                    dir_path = os.path.join(save_dir, save_path)

                    zipobj.extract(fs, dir_path)

                    prev_name = os.path.join(
                        dir_path, 'T1/T1_brain_to_MNI.nii.gz')
                    cur_name = os.path.join(
                        dir_path, f'T1/{save_path}_MNI.nii.gz')
                    os.rename(prev_name, cur_name)

                    # now move
                    final_path = os.path.join(
                        save_dir, f'{save_path}_MNI.nii.gz')
                    shutil.move(cur_name, final_path)

                    # finally remove the directory
                    shutil.rmtree(dir_path)
                    print(f"Extraction of {final_path} is completed")


def extract_t1_brain_gene(save_dir, file_type="T1/T1_brain_to_MNI.nii.gz", file_name_map=None, ukb_dir="<EXTERNAL: UKB T1/T2 zip source dir>"):
    '''
    Extract number_t1 of T1 brain from the UKB imaging release to save_dir. Individual
    folder is identified patientid_20252


    '''
    files = [p for p in os.listdir(ukb_dir) if '_20252_2_0.zip' in p]

    print(f"we are extracting {file_type} for total {len(files)} samples")
    for file in files:
        print(f"workig with {file}")
        with ZipFile(os.path.join(ukb_dir, file)) as zipobj:

            listOfFileNames = zipobj.namelist()

            for fs in listOfFileNames:
                if fs in file_type:
                    print(f"working with {fs}")
                    save_path = file.split(".")[0]

                    dir_path = os.path.join(save_dir, save_path)

                    zipobj.extract(fs, dir_path)

                    prev_name = os.path.join(dir_path, fs)
                    print(f"prev_name  {prev_name}")

                    cur_name = os.path.join(
                        dir_path, f'{"/".join(fs.split("/")[:-1])}/{save_path}_{file_name_map[fs]}')
                    os.rename(prev_name, cur_name)
                    print(f"cur_name  {cur_name}")
                    # now move
                    final_path = os.path.join(
                        save_dir, f'{save_path}_{file_name_map[fs]}')
                    print(f"final_path  {final_path}")
                    shutil.move(cur_name, final_path)

                    # finally remove the directory
                    shutil.rmtree(dir_path)
                    # print()
            print(f"Extraction of {file} is completed")


def drive_extract_t1_brain_gene():

    save_dir = cfg.embedding.t1_dir
    file_type = ['T1/T1_brain.nii.gz', 'T1/transforms/T1_to_MNI_linear.mat',
                 'T1/transforms/T1_to_MNI_warp_coef.nii.gz',
                 'T1/T1_fast/T1_brain_seg.nii.gz',
                 'T1/T1_first/T1_first_all_fast_firstseg.nii.gz'
                 ]
    file_name_map = {
        'T1/T1_brain_to_MNI.nii.gz': 'MNI.nii.gz',
        'T1/T1_unbiased_brain.nii.gz': 'T1_unbiased_brain.nii.gz',
        'T1/T1_brain.nii.gz': 'T1_brain.nii.gz',
        'T1/transforms/T1_to_MNI_linear.mat': 'T1_to_MNI_linear.mat',
        'T1/transforms/T1_to_MNI_warp_coef.nii.gz': 'T1_to_MNI_warp_coef.nii.gz',

        'T1/T1_fast/T1_brain_seg.nii.gz': 'T1_brain_seg.nii.gz',
        'T1/T1_first/T1_first_all_fast_firstseg.nii.gz': 'T1_first_all_fast_firstseg.nii.gz'
    }
    extract_t1_brain(save_dir=save_dir, file_type=file_type,
                     file_name_map=file_name_map, ukb_dir="<EXTERNAL: UKB T1/T2 zip source dir>")


def make_path(base_path, file_path, split="/"):
    p = base_path

    for temp_p in file_path.split(split):
        p = os.path.join(p, temp_p)

    return p


def convert_maks(mri_dir, save_dir, starts=0, ends=1000):
    # converting the maks to linear space

    p = mri_dir
    fls = os.listdir(p)

    save_dir = save_dir
    os.makedirs(save_dir, exist_ok=True)

    for folder in fls[starts:ends]:
        # get full folder path
        ful_folder_path = os.path.join(p, folder)

        # print(f"ful_folder_path:{ful_folder_path}")

        # get the transform matrix path

        tf_mat = make_path(
            ful_folder_path, 'T1/transforms/T1_to_MNI_linear.mat')

        # print(f"tf_mat:{tf_mat}")

        linear_mri_path = os.path.join(
            ful_folder_path, 'T1', 'T1_unbiased_brain_linear.nii.gz')

        # get mask paths from the file_name_map
        for mask in ['T1/T1_fast/T1_brain_seg.nii.gz', 'T1/T1_first/T1_first_all_fast_firstseg.nii.gz']:

            maks_name = mask.split("/")[-1].split(".")[0]
            mask_path = make_path(ful_folder_path, mask)
            # print(f"maks_path:{mask_path}")

            # get the output path.it should be in my folder

            linear_masks_output_path = os.path.join(
                save_dir, f"{folder}_linear_{maks_name}")
            # print(f"linear_masks_output_path:{linear_masks_output_path}")

            # execute command
            os.system(
                f"flirt -in {mask_path} -ref {linear_mri_path} -init {tf_mat} -interp nearestneighbour -applyxfm -out {linear_masks_output_path}")

        print(f"done mask conversion for {folder}")
    print("")


def driver_convert_maks(starts=0, ends=10000):

    mri_dir = cfg.embedding.t1_dir
    save_dir = os.path.join(
        config.data_path(), 'RBGWAS_RELATED_DATA', 'MASKS_MNI_LINEAR')
    os.makedirs(save_dir, exist_ok=True)
    # tot_file = 44614
    # starts= 0
    # ends = 1000
    convert_maks(mri_dir, save_dir, starts=starts, ends=ends)


# ------------------------ NON LINEAR MNI SPACE MASK CONVERSION-----------------------

def convert_masks_to_non_mni_space_per_folder(image_folder_id, image_dir="", template_path="", save_dir=""):

    folder_path = os.path.join(image_dir, image_folder_id)

    # get coefficient
    std_mni_coeff_path = os.path.join(
        folder_path, 'T1', 'transforms', 'T1_to_MNI_warp_coef.nii.gz')

    # subcortical_mask_path
    sub_cort_mask_path = os.path.join(
        folder_path, 'T1', 'T1_first', 'T1_first_all_fast_firstseg.nii.gz')

    # wmc path
    wmc_mask_path = os.path.join(
        folder_path, 'T1', 'T1_fast', 'T1_brain_seg.nii.gz')

    sub_cort_output_mask_path = os.path.join(
        save_dir, f"{image_folder_id}_T1_first_all_fast_firstseg_MNI.nii.gz")
    wmc_output_mask_path = os.path.join(
        save_dir, f"{image_folder_id}_T1_brain_seg_MNI.nii.gz")

    # covert the mask
    os.system(
        f"applywarp --ref={template_path} --in={sub_cort_mask_path} --out={sub_cort_output_mask_path} --warp={std_mni_coeff_path} --interp=nn")
    os.system(
        f"applywarp --ref={template_path} --in={wmc_mask_path} --out={wmc_output_mask_path} --warp={std_mni_coeff_path} --interp=nn")
    print(f"done for {image_folder_id}")


def driver_convert_masks_to_non_mni_space_per_folder(image_dir, template_path, save_dir):

    partial_convert = partial(convert_masks_to_non_mni_space_per_folder,
                              image_dir=image_dir,
                              template_path=template_path,
                              save_dir=save_dir)

    # get all the folders
    # all_folders = [f for f in os.listdir(image_dir) if ('_20252_2_0' in f) or ('_20252_3_0' in f)]
    all_folders = [f for f in os.listdir(image_dir) if '_20252_3_0' in f]
    print(
        f"there are {len(all_folders)} in the image directory at {image_dir}")
    print(f"the template is located at :{template_path}")
    print(f"all converted masks will be saved in :{save_dir}")

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.map(partial_convert, all_folders)

    pool.close()
    pool.join()
    print('Done')


# ------------------------ END OF NON LINEAR MNI SPACE MASK CONVERSION-----------------------

# creating subject dict

print(f"Loading data extraction script")


def extract_subjects_dicts(embedding_path, region_dict_save_dir, agg='mean'):

    # get the eid files name
    eid_files = os.listdir(embedding_path)

    # get all region name from a folder
    f = os.path.join(embedding_path, eid_files[0])
    region_embeddings_name = os.listdir(f)
    regions = set(["_".join(f.split(".")[0].split("_")[4:-1])
                  for f in region_embeddings_name])

    # iterate over region names to save the dictionary for each region

    for region in regions:
        # agg='mean'

        region_file_name = f"{region}_{agg}.pt"
        # find all files with this region names from all eids

        files = glob.glob(f"{embedding_path}/*/*{region_file_name}")
        indv_dicts = [{os.path.basename(f).split("_")[0]: ("_".join(
            os.path.basename(f).split("_")[:4]), torch.load(f).numpy())} for f in files]
        region_dict = reduce(lambda a, b: {**a, **b}, indv_dicts)

        save_name = os.path.join(region_dict_save_dir, f"{region}_{agg}.pkl")
        with open(save_name, 'wb') as f:
            pkl.dump(region_dict, f)

        print(f"{region}=>{save_name}=> {len(region_dict)}")


def combine_subject_dicts(subject_dict_dir, save_dir):
    ''' Combine dictionary of same regions from different cohorts. used to combine discovery, replicaition, remaining
        cohot region embedding.
        Arguments:
            subject_dict_dir: 
    '''

    os.makedirs(save_dir, exist_ok=True)

    regions_names = os.listdir(os.path.join(
        subject_dict_dir, os.listdir(subject_dict_dir)[0]))

    for regions_name in regions_names:
        print(f"Working with {regions_name}")
        files = glob.glob(f"{subject_dict_dir}/*/*{regions_name}")

        ls = []
        for file in files:
            with open(file, 'rb') as f:
                temp = pkl.load(f)
                ls.append(temp)
        region_dict = reduce(lambda a, b: {**a, **b}, ls)

        with open(os.path.join(save_dir, regions_name), 'wb') as f:
            pkl.dump(region_dict, f)
    print(f"Done")

# subject_dict_dir = os.path.join(config.data_path(),'RBGWAS_RELATED_DATA','SUBJECT_DICTS')
# save_dir = os.path.join(subject_dict_dir,'Combined_subject_dicts')


# combine_subject_dicts(subject_dict_dir,save_dir)
if __name__ == "__main__":

    # driver_wrapper_get_embedding_all_region()
    '''
    grand_root_path = "<EXTERNAL: project root for embedding extraction>"
    warp_field_p = os.path.join(grand_root_path,'region_dicts.pkl')
    region_template = os.path.join(grand_root_path,'HarvardOxford-sub-maxprob-thr50-1mm.nii.gz')
    # Khush's folder MRI
    #cpu_1
    sample_dict_name = 'mni_img.pkl'
    start_index = 0
    end_index = 10986
    device = 'cuda:1'

    wrapper_get_extraction(start_index, end_index, grand_root_path,region_template, sample_dict_name,device)
    '''

    '''
    #ziqian's folder
    #cpu_2
    sample_dict_name = 'ziqian_t1t2_sample_dict.pkl'
    
    start_index = 0
    end_index = 10000
    
    device = 'cuda:2'
    wrapper_get_extraction(start_index, end_index, grand_root_path,sample_dict_name, device)
    
    #cpu_3
    start_index = 10000
    end_index = 20000
    device = 'cuda:3'
    wrapper_get_extraction(start_index, end_index, grand_root_path,sample_dict_name, device)

    #cpu_4
    start_index = 20000
    end_index = 31806
    
    device = 'cuda:4'
    wrapper_get_extraction(start_index, end_index, grand_root_path,sample_dict_name, device)
    '''
    '''
    save_dir = "<EXTERNAL: T1 MNI output dir>"
    extract_t1_brain(save_dir,file_type = "T1/T1_brain_to_MNI.nii.gz",ukb_dir = "<EXTERNAL: UKB T1/T2 zip source dir>")
    '''
    # driver_convert_maks()
