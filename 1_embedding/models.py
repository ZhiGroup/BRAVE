import copy
import torch.nn.init as init
from GPUtil import showUtilization as gpu_usage
import time
import numpy as np
import random
import itertools
from functools import reduce
from operator import mul

import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR


import lightning.pytorch as pl
import os
import nibabel as nib
from zipfile import ZipFile
import random

try:

    from . import config

except:

    import config


def conv1X1_3d(in_planes, out_planes):
    '''
        for instance loss
    '''
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=True)


class MLP3d(pl.LightningModule):
    def __init__(self, in_dim, inner_dim=4096, out_dim=256):
        super(MLP3d, self).__init__()
        self.linear1 = conv1X1_3d(in_dim, inner_dim)
        self.bn1 = nn.BatchNorm3d(inner_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = conv1X1_3d(inner_dim, out_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        return x


def Proj_Head(in_dim=256, inner_dim=4096, out_dim=256):
    return MLP3d(in_dim, inner_dim, out_dim)


def Pred_Head(in_dim=256, inner_dim=4096, out_dim=256):
    return MLP3d(in_dim, inner_dim, out_dim)


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


class BasicBlock(pl.LightningModule):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        # print(f"x: {x.shape}")
        # print(f"x.device {x.device}")
        # print(f"self.conv1.device{self.conv1.weight.device}")
        out = self.conv1(x)  # (n,c,a//2,b//2,c//2), as stride of 2 was used
        out = self.bn1(out)
        out = F.relu(out)

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
        out = F.relu(out)

        return out


class Bottleneck(pl.LightningModule):
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


class Resnet18(pl.LightningModule):
    def __init__(self, block, block_inplanes, layers, n_input_channels=1, conv1_t_size=3, conv1_t_stride=1,
                 apply_z_stride=[2, 1, 1, 2],  no_max_pool=True, shortcut_type='B'):
        super(Resnet18, self).__init__()
        self.block = block
        self.planes = 64
        self.blocks = 2
        self.stride = 1
        self.in_planes = 64

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(7, conv1_t_size, 7),
                               stride=(1, conv1_t_stride, 1),
                               padding=(3, conv1_t_size // 2, 3),
                               bias=False)

        # self.conv2_x = self._make_layer(self.block, self.planes, self.blocks, self.stride)
        self.conv2_x = self._make_layer(block,
                                        block_inplanes[0],
                                        layers[0],
                                        shortcut_type,
                                        conv_type="B",
                                        stride=2,
                                        apply_z_stride=apply_z_stride[0]
                                        )
        self.conv3_x = self._make_layer(block,
                                        block_inplanes[1],
                                        layers[1],
                                        shortcut_type,
                                        conv_type="B",

                                        stride=2,
                                        apply_z_stride=apply_z_stride[1]
                                        )

        self.conv4_x = self._make_layer(block,
                                        block_inplanes[2],
                                        layers[2],
                                        shortcut_type,
                                        conv_type="B",

                                        stride=2,
                                        apply_z_stride=apply_z_stride[2]
                                        )
        self.conv5_x = self._make_layer(block,
                                        block_inplanes[3],
                                        layers[3],
                                        shortcut_type,
                                        conv_type="B",

                                        stride=2,
                                        apply_z_stride=apply_z_stride[3]
                                        )

    def _make_layer(self, block, planes, blocks, shortcut_type, conv_type="A", stride=1, apply_z_stride=1):
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
                      downsample=downsample))
        else:
            downsample = nn.Sequential(
                conv1x1x1(self.in_planes, planes * block.expansion,
                          [stride, stride, apply_z_stride]),
                nn.BatchNorm3d(planes * block.expansion))
            layers.append(
                block(in_planes=self.in_planes,
                      planes=planes,
                      stride=[stride, stride, apply_z_stride],
                      downsample=downsample))

        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x = [32,96,96] or x = [a*32, b*32, c*32] , if x = [32,96,96] then a = 1, b = 3,c=3
        x = self.conv1(x)  # [32,96,96] =>  [a*32, b*32, c*32]
        c2 = self.conv2_x(x)  # [16,48,48] => [a*32/2, b*32//2, c*32//2]
        c3 = self.conv3_x(c2)  # [16,24,24] => [a*32/2, b*32//4, c*32//4]
        c4 = self.conv4_x(c3)  # [16,12,12] => [a*32/2, b*32//8, c*32//8]
        c5 = self.conv5_x(c4)  # [8,6,6] => [a*32/4, b*32//16, c*32//16]

        return (c2, c3, c4, c5)


def generate_model(apply_z_stride=[2, 1, 1, 2]):
    block = BasicBlock
    block_inplanes = [64, 128, 256, 512]
    layers = [2, 2, 2, 2]
    return Resnet18(block=BasicBlock, block_inplanes=block_inplanes, layers=layers, apply_z_stride=apply_z_stride)


class FPN(pl.LightningModule):
    def __init__(self, apply_z_stride=[2, 1, 1, 2]):
        super().__init__()
        model_depth = 18
        self.backbone_resnet = generate_model(
            apply_z_stride=apply_z_stride)  # return tensors in gpu_3

        # Smooth layers
        self.smooth2 = conv3x3x3(128, 128)
        # self.bn_smooth2 = nn.BatchNorm3d(128)
        self.smooth5 = conv3x3x3(128, 128)
        # self.bn_smooth5 = nn.BatchNorm3d(128)

        # Lateral layers

        self.latlayer5 = conv1x1x1(512, 128)
        # self.bn_latlayer5 = nn.BatchNorm3d(128)

        self.latlayer4 = conv1x1x1(256, 128)
        # self.bn_latlayer4 = nn.BatchNorm3d(128)

        self.latlayer3 = conv1x1x1(128, 128)
        # self.bn_latlayer3 = nn.BatchNorm3d(128)

        self.latlayer2 = conv1x1x1(64, 128)
        # self.bn_latlayer2 = nn.BatchNorm3d(128)

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
        # print(f"x: {x.shape}")
        b, c, img_D, img_W, img_H = x.shape
        # top down flow
        c2, c3, c4, c5 = self.backbone_resnet(x)

        '''
        print("before applying lateral connection")
        
        print(f"c2 => {c2.shape}")
        print(f"c3 => {c3.shape}")
        print(f"c4 => {c4.shape}")
        print(f"c5 => {c5.shape}")
        print()
        '''
        # lateral connection

        c5 = self.latlayer5(c5)
        # c5 = self.bn_latlayer5(c5)

        c4 = self.latlayer4(c4)
        # c4 = self.bn_latlayer4(c4)

        c3 = self.latlayer3(c3)
        # c3 = self.bn_latlayer3(c3)

        c2 = self.latlayer2(c2)
        # c2 = self.bn_latlayer2 (c2)

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
        # shape [batch, embed_dim, original_patch_shape//16]
        global_embedding = self.smooth5(c5)
        # global_embedding = self.bn_smooth5(global_embedding)
        # print(f"global_embedding => {global_embedding.shape}")

        # this normalizatio is wrong. we need to normalize the embedding dimension only.
        # global_embedding = F.normalize(global_embedding,p=2,dim= (2,3,4))

        # the correct one is : comment it when do inference-------------------------------
        # global_embedding = F.normalize(global_embedding,p=2,dim= 1)
        # assert torch.norm(global_embedding,dim=1).mean() == 1.
        # print(f"global_embedding after L2 normalization => {global_embedding.shape}")

        # shape [batch, embed_dim, original_patch_shape//2]
        local_embedding = self.smooth2(p2)
        # local_embedding = self.bn_smooth2(local_embedding)
        # print(f"local_embedding => {local_embedding.shape}")

        # this normalizatio is wrong. we need to normalize the embedding dimension only.
        # local_embedding = F.normalize(local_embedding,p=2,dim= (2,3,4))
        # the correct one is .comment it when do inference-------------------------------
        # local_embedding = F.normalize(local_embedding,p=2,dim= 1)
        # assert torch.norm(local_embedding,dim=1).mean() == 1.
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
        return local_embedding, global_embedding


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_model(args, model_type='vanila'):

    model_kwargs = {'batch_size': args['batch_size'],
                    'n_pos_voxel': args['n_pos_voxel'],
                    'patch_size': args['patch_size'],
                    'apply_z_stride': args['apply_z_stride'],
                    'embedding_dim': args['embedding_dim'],
                    'apply_z_stride': args['apply_z_stride'],
                    'los_temp': args['los_temp'],
                    'device': args['device'],
                    'radius': args['radius'],
                    'lr': args['lr'],
                    'weight_decay': args['weight_decay'],

                    'factor_n_negative_samples': args['factor_n_negative_samples'],
                    'pct_start': args['pct_start'],
                    'div_factor': args['div_factor'],
                    'final_div_factor': args['final_div_factor'],
                    'num_instances': args['num_instances'],
                    'max_epoch': args['max_epoch'],
                    'los_temp_instance': args['los_temp_instance'],
                    'momentum': args['momentum'],
                    'memory_bank_size': args['memory_bank_size'],
                    'reconstruction_loss': args['reconstruction_loss']
                    }
    if model_type == 'vanila':
        model = Model(model_kwargs)
    else:
        model = Model_Momentum(model_kwargs)
    return model


class Model(pl.LightningModule):
    '''Base Model. Also infoNCE loss is implemented in Forward function. So, it is also m4_exp_3
    '''

    def __init__(self, model_kwargs):
        super().__init__()
        self.model_kwargs = model_kwargs
        print(f"--------------------Model Instantiated with followings-------------------")
        print(self.model_kwargs)
        self.apply_z_stride = self.model_kwargs['apply_z_stride'] if 'apply_z_stride' in self.model_kwargs else [
            2, 1, 1, 2]

        self.n_batch = self.model_kwargs['batch_size']
        self.n_pos_voxel = self.model_kwargs['n_pos_voxel']
        self.patch_size = self.model_kwargs['patch_size']

        self.embedding_dim = self.model_kwargs['embedding_dim']
        self.lr = self.model_kwargs['lr']
        self.los_temp = self.model_kwargs['los_temp']

        self.weight_decay = self.model_kwargs['weight_decay']
#        self.device = self.model_kwargs['device'] if 'device' in self.model_kwargs else torch.device(
#            'cpu')
        self.radius = self.model_kwargs['radius'] if 'radius' in self.model_kwargs else 1
        if sum(self.model_kwargs['apply_z_stride']) == 6:
            self.model_kwargs['equal_stride'] = False
        else:
            self.model_kwargs['equal_stride'] = True
        print(f"apply_z_stride:{self.apply_z_stride}")
        print(f"equal_stride:{self.model_kwargs['equal_stride']}")

        self.factor_n_negative_samples = self.model_kwargs[
            'factor_n_negative_samples'] if 'factor_n_negative_samples' in self.model_kwargs else 40
        print(f"factor_n_negative_samples:{self.factor_n_negative_samples}")
        # add mlp projection head

        self.scaling_factor = (
            self.patch_size[0]*self.patch_size[1]*self.patch_size[2])/(96*96*96)

        if reduce(mul, self.apply_z_stride, 1) == 16:
            self.global_n_neg = 190  #
        else:
            # valid for [2,1,1,2] z-stride
            self.global_n_neg = int(500*(self.scaling_factor))

        self.local_cand_neg = int(20000*(self.scaling_factor))

        self.global_n_random = self.global_n_neg*(self.n_batch-1)
        self.save_hyperparameters()
        print(f"Patch Size:{self.patch_size}")
        print(f"Batch Size:{self.n_batch}")
        print(f"scaling_factor:{self.scaling_factor}")
        print(f"n_pos_voxel:{self.n_pos_voxel}")
        print(f"global_n_neg:{self.global_n_neg}")
        print(f"embedding_dim:{self.embedding_dim}")
        print(f"local_cand_neg:{self.local_cand_neg}")
        print(
            f"local_neg:{self.local_cand_neg//self.model_kwargs['factor_n_negative_samples']}")
        print(f"global_n_random:{self.global_n_random}")
        print(f"lr: {self.lr}")

        self.fpn = FPN(apply_z_stride=self.apply_z_stride)

        self.instance_project_head_local = Proj_Head(
            in_dim=128, inner_dim=128, out_dim=128)
        # self.instance_pred_head_local = Pred_Head( in_dim=128, inner_dim=4096, out_dim=128 )

        self.instance_project_head_global = Proj_Head(
            in_dim=128, inner_dim=128, out_dim=128)
        # self.instance_pred_head_global = Pred_Head( in_dim=128, inner_dim=4096, out_dim=128 )
        self.avgpool_local = nn.AvgPool3d(48, stride=1)
        self.avgpool_global = nn.AvgPool3d(6, stride=1)

        self.projection = nn.Sequential(
            nn.Linear(in_features=128, out_features=5*128),
            nn.BatchNorm1d(5*128),
            nn.ReLU(),
            nn.Linear(in_features=5*128, out_features=128),
            nn.BatchNorm1d(128),
        )

    def info_nce_instance(self, view_local_1, view_global_1, view_local_2, view_global_2,
                          temperature=0.5, global_loss_wt=0.5):
        # local,global : (nb, embed_dim)

        # nb, embed_dim) @nb, embed_dim) -> (nb, nb)
        sim_matrix_local = torch.matmul(
            view_local_1, view_local_2.T)/temperature
        sim_matrix_global = torch.matmul(
            view_global_1, view_global_2.T)/temperature

        # each row in the sim_matrix_local is teated as differnt sample, with nb class, diagonal values are true class logscore

        labels = torch.arange(view_local_1.size(0)).to(view_local_1.device)

        loss_local = torch.nn.CrossEntropyLoss()(sim_matrix_local, labels)
        loss_global = torch.nn.CrossEntropyLoss()(sim_matrix_global, labels)

        loss = loss_local*(1-global_loss_wt) + loss_global*global_loss_wt
        return loss

    def get_instance_embedding(self, local_1, global_1, local_2, global_2):
        # local,globa: (nb, embed_dim, w,h,d)
        proj_instance_local_1 = self.instance_project_head_local(local_1)
        # pred_instacne_local_1 = self.instance_pred_head_local(proj_instance_local_1)
        pred_instance_local_1 = F.normalize(self.avgpool_local(
            proj_instance_local_1).view(proj_instance_local_1.size(0), -1), dim=1)

        proj_instance_local_2 = self.instance_project_head_local(local_2)
        # pred_instacne_local_2 = self.instance_pred_head_local(proj_instance_local_2)
        pred_instance_local_2 = F.normalize(self.avgpool_local(
            proj_instance_local_2).view(proj_instance_local_2.size(0), -1), dim=1)

        proj_instance_global_1 = self.instance_project_head_global(global_1)
        # proj_instance_global_1 = self.instance_pred_head_global(proj_instance_global_1)
        pred_instance_global_1 = F.normalize(self.avgpool_global(
            proj_instance_global_1).view(proj_instance_global_1.size(0), -1), dim=1)

        proj_instance_global_2 = self.instance_project_head_global(global_2)
        # proj_instance_global_2 = self.instance_pred_head_global(proj_instance_global_2)
        pred_instance_global_2 = F.normalize(self.avgpool_global(
            proj_instance_global_2).view(proj_instance_global_2.size(0), -1), dim=1)

        return pred_instance_local_1, pred_instance_global_1, pred_instance_local_2, pred_instance_global_2

    def get_local_global_anchors(self, local_embedding_1st_ls, local_embedding_2nd_ls,
                                 global_embedding_1st_ls, global_embedding_2nd_ls, pos, n_batch):

        # we need to divide the coordinates of to z/4 axis and x,y axis by 16 (global embeddings)
        # each last dimension in pos_z contains the coordinates of achor  coordinates : (0,1,2)th indexes, and the paired one is (3,4,5)th indexes
        # z dimension is the last dimension. so z dimension would 2,5th idexes
        # for global embedding, # shape (nb, n_pos_voxel,6)
        pos_z = pos.clone().to(pos.device)
        if self.model_kwargs['equal_stride']:
            pos_z[:, :, [0, 1, 2, 3, 4, 5]] = (pos_z[:, :, [
                                               0, 1, 2, 3, 4, 5]]//torch.tensor([16], dtype=torch.long).to(pos.device)).to(torch.long)
        else:
            pos_z[:, :, [2, 5]] = (
                pos_z[:, :, [2, 5]]//torch.tensor([4], dtype=torch.long).to(pos.device)).to(torch.long)
            pos_z[:, :, [0, 1, 3, 4]] = (pos_z[:, :, [
                                         0, 1, 3, 4]]//torch.tensor([16], dtype=torch.long).to(pos.device)).to(torch.long)

        # shape pos_z # shape (nb, n_pos_voxel,6)

        # we need to divide the all coordinates by 2 ( local embedding)
        pos_xy = pos.clone().to(pos.device)  # shape (nb, n_pos_voxel,6)
        # shape (nb, n_pos_voxel,6)
        pos_xy = (
            pos_xy//torch.tensor([2], dtype=torch.long).to(pos.device)).to(torch.long)

        # global
        anchor_global_1st_ls = global_embedding_1st_ls[torch.arange(n_batch).reshape(n_batch, -1).long(), :, pos_z[torch.arange(
            n_batch).long(), :, 0], pos_z[torch.arange(n_batch).long(), :, 1], pos_z[torch.arange(n_batch).long(), :, 2]]
        anchor_global_2nd_ls = global_embedding_2nd_ls[torch.arange(n_batch).reshape(n_batch, -1).long(), :, pos_z[torch.arange(
            n_batch).long(), :, 3], pos_z[torch.arange(n_batch).long(), :, 4], pos_z[torch.arange(n_batch).long(), :, 5]]
        # shape: anchor_global_1st_ls,anchor_global_2nd_ls : (nb, n_pos_voxel, embedding_dim)

        # local
        anchor_local_1st_ls = local_embedding_1st_ls[torch.arange(n_batch).reshape(n_batch, -1).long(), :, pos_xy[torch.arange(
            n_batch).long(), :, 0], pos_xy[torch.arange(n_batch).long(), :, 1], pos_xy[torch.arange(n_batch).long(), :, 2]]
        anchor_local_2nd_ls = local_embedding_2nd_ls[torch.arange(n_batch).reshape(n_batch, -1).long(), :, pos_xy[torch.arange(
            n_batch).long(), :, 3], pos_xy[torch.arange(n_batch).long(), :, 4], pos_xy[torch.arange(n_batch).long(), :, 5]]

        return (anchor_global_1st_ls, anchor_global_2nd_ls, anchor_local_1st_ls, anchor_local_2nd_ls), (pos_z)

    def get_similarity_tensors(self, anchor_global_1st_ls, global_embedding_1st_ls, global_embedding_2nd_ls,
                               anchor_local_1st_ls, local_embedding_1st_ls, local_embedding_2nd_ls):

        # step 7: calculate similarity maps sg, sg',sl,sl'

        # sg would be the similarity map of anchor_global_1st_ls with global_embedding_1st_ls
        sg = (torch.einsum('bij, bjklm-> biklm',
              [anchor_global_1st_ls, global_embedding_1st_ls])/torch.tensor(128).sqrt()).to(anchor_global_1st_ls.device)
        # sg' would be the similarity map of anchor_global_1st_ls with global_embedding_2nd_ls
        sg_prime = (torch.einsum('bij, bjklm-> biklm',
                    [anchor_global_1st_ls, global_embedding_2nd_ls])/torch.tensor(128).sqrt()).to(anchor_global_1st_ls.device)
        # shape : sg,sg_prime:
        # (nb, n_pos_voxel, patche_shape//16) or (nb, n_pos_voxel, patche_shape[0]//16, patche_shape[0]//16,patche_shape[0]//4) depend of apply_z_stride

        # lg would be the similarity map of anchor_local_1st_ls with local_embedding_1st_ls
        sl = (torch.einsum('bij, bjklm-> biklm',
              [anchor_local_1st_ls, local_embedding_1st_ls])/torch.tensor(128).sqrt()).to(anchor_local_1st_ls.device)
        # sg' would be the similarity map of anchor_global_1st_ls with global_embedding_2nd_ls
        sl_prime = (torch.einsum('bij, bjklm-> biklm',
                    [anchor_local_1st_ls, local_embedding_2nd_ls])/torch.tensor(128).sqrt()).to(anchor_local_1st_ls.device)
        # shape: sl,sl_prime: (nb, n_pos_voxel, patche_shape//2)

        return (sg, sg_prime, sl, sl_prime)

    def get_global_negative_voxels(self, sg, sg_prime, pos_z, global_embedding_1st_ls,
                                   global_embedding_2nd_ls, n_batch, b):

        # set similarity of positive voxel with themselves to zero

        sg[torch.arange(n_batch).reshape(n_batch, -1).long().to(sg.device), torch.arange(self.n_pos_voxel).long().to(sg.device), pos_z[torch.arange(n_batch).long().to(sg.device),
                                                                                                                                       :, 0], pos_z[torch.arange(n_batch).long().to(sg.device), :, 1], pos_z[torch.arange(n_batch).long().to(sg.device), :, 2]] = torch.tensor(-float("inf")).to(sg.device)

        sg_prime[torch.arange(n_batch).reshape(n_batch, -1).long().to(sg_prime.device), torch.arange(self.n_pos_voxel).long().to(sg_prime.device), pos_z[torch.arange(n_batch).long().to(
            sg_prime.device), :, 3], pos_z[torch.arange(n_batch).long().to(sg_prime.device), :, 4], pos_z[torch.arange(n_batch).long().to(sg_prime.device), :, 5]] = torch.tensor(-float("inf")).to(sg_prime.device)

        # shape: sg,sg_prime
        # (nb, n_pos_voxel, patche_shape//16) or #
        # (nb, n_pos_voxel, patche_shape[0]//16, patche_shape[0]//16,patche_shape[0]//4) depend of apply_z_stride

        # sort the sg,sg_prime, sl,sl_prime
        # (nb, n_pos_voxel, product of (patch_shape//16))
        sg = sg.view(b, self.n_pos_voxel, -1)
        # (nb, n_pos_voxel, product of (patch_shape//16))
        sg_prime = sg_prime.view(b, self.n_pos_voxel, -1)
        _, indices_g = torch.sort(sg, descending=True)
        # (nb, n_pos_voxel, product of (patch_shape//16)), But last time,checked not.
        # TODO. we will log the indices_g, indices_g_prime with coordinates of positive voxels to see which
        # distribution of distace of negative samples coordinate from the positive samples coordinate
        _, indices_g_prime = torch.sort(sg_prime, descending=True)

        # (nb, n_pos_voxel, self.global_n_neg)
        indices_g = indices_g[..., :self.global_n_neg]
        # (nb, n_pos_voxel, self.global_n_neg)
        indices_g_prime = indices_g_prime[..., :self.global_n_neg]
        # indices_g.shape -> torch.Size([n_batch, self.self.n_pos_voxel, global_n_neg])
        # store_ls.append(indices_g.clone().cpu())
        # store_ls.append(indices_g_prime.clone().cpu())

        global_embedding_1st_ls_flat = global_embedding_1st_ls.reshape(
            n_batch, self.embedding_dim, -1)
        global_embedding_2nd_ls_flat = global_embedding_2nd_ls.reshape(
            n_batch, self.embedding_dim, -1)
        # shape: global_embedding_1st_ls_flat,global_embedding_2nd_ls_flat
        # (nb, embedding dim,  product of patche_shape//16)

        # global_embedding_1st_ls_flat.shape -> (n_batch,self.self.n_pos_voxel,self.self.embedding_dim,dim_x,dim_z*dim_y)

        global_embedding_1st_ls_flat = global_embedding_1st_ls_flat.unsqueeze(
            1).expand(-1, self.n_pos_voxel, -1, -1)
        global_embedding_2nd_ls_flat = global_embedding_2nd_ls_flat.unsqueeze(
            1).expand(-1, self.n_pos_voxel, -1, -1)
        # shape: global_embedding_1st_ls_flat,global_embedding_2nd_ls_flat
        # (nb, n_pos_voxel, embedding_dim,product of patche_shape//16)

        # our target global hard negative sample should be [3,4,128,10]
        indices_g = indices_g.unsqueeze(
            2).expand(-1, -1, self.embedding_dim, -1)
        # (n_batch,self.n_pos_voxel,self.embedding_dim,self.global_n_neg)
        indices_g_prime = indices_g_prime.unsqueeze(
            2).expand(-1, -1, self.embedding_dim, -1)
        # indices_g,indices_g_prime
        # (nb,n_pos_voxel, embedding dim, global_n_neg)

        neg_global_emb_g = torch.gather(
            global_embedding_1st_ls_flat, dim=-1, index=indices_g)
        neg_global_emb_g_prime = torch.gather(
            global_embedding_2nd_ls_flat, dim=-1, index=indices_g_prime)
        # neg_global_emb_g,neg_global_emb_g_prime
        # (nb,n_pos_voxel, embedding dim, global_n_neg)

        # del global_embedding_1st_ls_flat
        # del global_embedding_2nd_ls_flat
        # neg_global_emb_g_prime -> (n_batch,self.n_pos_voxel,self.self.embedding_dim,self.global_n_neg)

        # concat global hard embedding from sg,sg_prime and randomly choose self_global_n_neg
        # (nb,n_pos_voxel, embedding dim, 2*global_n_neg)
        neg_global_emb = torch.cat(
            [neg_global_emb_g, neg_global_emb_g_prime], dim=-1)
        # neg_global_emb_g,neg_global_emb_g_prime
        # del neg_global_emb_g
        # del neg_global_emb_g_prime
        # neg_global_emb.shape -> ((n_batch,self.n_pos_voxel,self.self.embedding_dim,2*self.global_n_neg))
        neg_glbal_f_embed = torch.index_select(neg_global_emb, dim=-1, index=(
            torch.randperm(2*self.global_n_neg)[:self.global_n_neg]).to(neg_global_emb.device))
        # shape : neg_glbal_f_embed:(n_batch,self.n_pos_voxel,self.embedding_dim,self.global_n_neg)

        return neg_glbal_f_embed

    def get_voxel_coordinates_for_local_negative_voxel_selection(self, pos, W, H, D):
        # define the radius boundary
        left_pos_x, left_pos_y, left_pos_z = pos[:, :, [
            0, 0+3]]-self.radius, pos[:, :, [1, 1+3]]-self.radius, pos[:, :, [2, 2+3]]-self.radius

        # shape: all of the abover left_* : (nb, n_pos_voxel, 2),
        # last dim,0=> left edge of cicle around the positive voxel in first patch
        # last dim,1=> left edge of cicle around the positive voxel in 2nd  patch

        right_pos_x, right_pos_y, right_pos_z = pos[:, :, [
            0, 0+3]]+self.radius, pos[:, :, [1, 1+3]]+self.radius, pos[:, :, [2, 2+3]]+self.radius

        #  if the boundary shoot over (beyond the shape of the )
        left_pos_x, left_pos_y, left_pos_z = torch.where(left_pos_x < 0, 0, left_pos_x), torch.where(
            left_pos_y < 0, 0, left_pos_y), torch.where(left_pos_z < 0, 0, left_pos_z)

        # this w,h,d are not i think, the highest coords are the edges of MRI.
        right_pos_x, right_pos_y, right_pos_z = torch.where(right_pos_x >= W, W-1, right_pos_x), torch.where(
            right_pos_y >= H, H-1, right_pos_y), torch.where(right_pos_z >= D, D-1, right_pos_z)
        # shape: all of the abover right_* : (nb, n_pos_voxel, 2)
        # now scale down to match the scaling done in the network
        # local embedding
        left_pos_x = left_pos_x.to(
            pos.device)//torch.tensor([2], dtype=torch.long).to(pos.device).to(torch.long)
        left_pos_z = left_pos_z.to(
            pos.device)//torch.tensor([2], dtype=torch.long).to(pos.device).to(torch.long)
        left_pos_y = left_pos_y.to(
            pos.device)//torch.tensor([2], dtype=torch.long).to(pos.device).to(torch.long)

        right_pos_x = right_pos_x.to(
            pos.device)//torch.tensor([2], dtype=torch.long).to(pos.device).to(torch.long)
        right_pos_z = right_pos_z.to(
            pos.device)//torch.tensor([2], dtype=torch.long).to(pos.device).to(torch.long)
        right_pos_y = right_pos_y.to(
            pos.device)//torch.tensor([2], dtype=torch.long).to(pos.device).to(torch.long)

        return (left_pos_x, left_pos_z, left_pos_y), (right_pos_x, right_pos_z, right_pos_y)

    def get_hard_local_negative_voxels(self, sl, sl_prime, sg_local, sg_prime_local,
                                       local_embedding_1st_ls, local_embedding_2nd_ls,
                                       n_batch, pos, W, H, D, b):

        sl = sl+sg_local  # (nb,n_pos_voxel, patch_shap//2)
        sl_prime = sl_prime + sg_prime_local

        # instead of masking only the positive voxel, mask all the voxels in the radius

        (left_pos_x, left_pos_z, left_pos_y), (right_pos_x, right_pos_z,
                                               right_pos_y) = self.get_voxel_coordinates_for_local_negative_voxel_selection(pos, W, H, D)
        for ib in range(n_batch):
            for ip in range(self.n_pos_voxel):
                # sl

                sl[ib, ip, left_pos_x[ib, ip, 0].item():right_pos_x[ib, ip, 0].item()+1,
                   left_pos_y[ib, ip, 0].item():right_pos_y[ib, ip, 0].item()+1,
                   left_pos_z[ib, ip, 0].item():right_pos_z[ib, ip, 0].item()+1] = torch.tensor(-float("inf"))

                # sl_prime
                sl_prime[ib, ip, left_pos_x[ib, ip, 1].item():right_pos_x[ib, ip, 1].item()+1,
                         left_pos_y[ib, ip, 1].item():right_pos_y[ib, ip, 1].item()+1,
                         left_pos_z[ib, ip, 1].item():right_pos_z[ib, ip, 1].item()+1] = torch.tensor(-float("inf"))

        # TODO: if radius =1, then there is no circle, only the positive voxel's similarity score is -inf
        # try with larger radius like 3, done in config file
        # (nb,n_pos_voxels, product of patch shape//2)
        sl = sl.view(b, self.n_pos_voxel, -1)
        sl_prime = sl_prime.view(b, self.n_pos_voxel, -1)
        # shape: sl,sl_prime (nb,n_pos_voxel, product of (patch_shape//2))

        # shape: indices_sgl,indices_sgl (nb,n_pos_voxel, product of (patch_shape//2))
        _, indices_sgl = torch.sort(sl, descending=True)
        _, indices_sgl_prime = torch.sort(sl_prime, descending=True)

        # del sl
        # del sl_prime

        # del sg
        # del sg_prime

        # shape: indices_sgl,indices_sgl (nb,n_pos_voxel, local_cand_neg)
        indices_sgl = indices_sgl[..., :self.local_cand_neg //
                                  self.factor_n_negative_samples]
        # shape: indices_sgl,indices_sgl (nb,n_pos_voxel, local_cand_neg)
        indices_sgl_prime = indices_sgl_prime[...,
                                              :self.local_cand_neg//self.factor_n_negative_samples]
        # store_ls.append(indices_sgl.clone().cpu())
        # store_ls.append(indices_sgl_prime.clone().cpu())
        # indices_sgl.shape -> (n_batch,self.self.n_pos_voxel,self.local_cand_neg)
        local_embedding_1st_ls_flat = local_embedding_1st_ls.reshape(
            b, self.embedding_dim, -1)
        local_embedding_2nd_ls_flat = local_embedding_2nd_ls.reshape(
            b, self.embedding_dim, -1)
        # shape:local_embedding_*_ls_flat: (nb, n_pos_voxel, product of patch_shape//2)
        local_embedding_1st_ls_flat = local_embedding_1st_ls_flat.unsqueeze(
            1).expand(-1, self.n_pos_voxel, -1, -1)
        local_embedding_2nd_ls_flat = local_embedding_2nd_ls_flat.unsqueeze(
            1).expand(-1, self.n_pos_voxel, -1, -1)

        # shape: local_embedding_*_ls_flat.shape -> (n_batch,self.n_pos_voxel,self.embedding_dim,product of patch_shape//2)

        # (nb,n_pos_voxel, embedding_dim, local_cand_neg)
        indices_sgl = indices_sgl.unsqueeze(
            2).expand(-1, -1, self.embedding_dim, -1)
        indices_sgl_prime = indices_sgl_prime.unsqueeze(
            2).expand(-1, -1, self.embedding_dim, -1)
        # indices_sgl_prime -> (n_batch,self.n_pos_voxel,self.self.embedding_dim,self.local_cand_neg)

        # (nb,n_pos_voxel, embedding_dim, local_cand_neg)
        neg_local_emb_l = torch.gather(
            local_embedding_1st_ls_flat, dim=-1, index=indices_sgl)
        neg_local_emb_l_prime = torch.gather(
            local_embedding_2nd_ls_flat, dim=-1, index=indices_sgl_prime)
        # del local_embedding_2nd_ls_flat
        # del local_embedding_1st_ls_flat
        # concate local_hard_negative from sl,sl_prime
        # shape :neg_local_emb_l,neg_local_emb_l_prime (n_batch,self.n_pos_voxel,self.local_cand_neg)
        neg_local_emb = torch.cat(
            [neg_local_emb_l, neg_local_emb_l_prime], dim=-1)
        # shape: neg_local_emb :(n_batch,self.n_pos_voxel,self.embedding_dim,2*self.local_cand_neg)

        # del neg_local_emb_l
        # del neg_local_emb_l_prime

        neg_local_f_embed = torch.index_select(neg_local_emb, dim=-1, index=(torch.randperm(
            2*(self.local_cand_neg//self.factor_n_negative_samples))[:(self.local_cand_neg//self.factor_n_negative_samples)]).to(neg_local_emb.device))

        return neg_local_f_embed

    def get_across_global_negative_voxels(self, global_embedding_1st_ls,
                                          global_embedding_2nd_ls,
                                          n_batch, dtype_):

        global_random_embed = torch.empty(
            (n_batch, self.n_pos_voxel, self.embedding_dim, self.global_n_random), dtype=dtype_, device=global_embedding_1st_ls.device)
        for bi in range(n_batch):
            for pi in range(self.n_pos_voxel):
                # excluding the sample of current positive voxel
                candidate_samples = torch.cat(
                    [global_embedding_1st_ls[0:bi], global_embedding_1st_ls[bi+1:]], dim=0)
                # shape: candidate_samples (n_batch-1,self.n_pos_voxel, patch_shape//16 ) if appy_equal_stried ==True
                candidate_samples = candidate_samples.permute(1, 0, 2, 3, 4).reshape(
                    self.embedding_dim, -1)  # (embedding_dim, n_batch*product of patch//16)
                selected_embeds = torch.index_select(candidate_samples, dim=-1, index=(torch.randperm(
                    candidate_samples.shape[-1])[:self.global_n_random]).to(global_embedding_1st_ls.device))  # (embedding_dim, global_n_random)

                # global_prime
                candidate_samples_prime = torch.cat(
                    [global_embedding_2nd_ls[0:bi], global_embedding_2nd_ls[bi+1:]], dim=0)
                # shape: candidate_samples (n_batch-1,self.n_pos_voxel, patch_shape//16 ) if appy_equal_stried ==True

                candidate_samples_prime = candidate_samples_prime.permute(
                    1, 0, 2, 3, 4).reshape(self.embedding_dim, -1).to(global_embedding_2nd_ls.device)
                selected_embeds_prime = torch.index_select(candidate_samples_prime, dim=-1, index=(
                    torch.randperm(candidate_samples_prime.shape[-1])[:self.global_n_random]).to(global_embedding_2nd_ls.device))

                # print(selected_embeds.shape)
                agg_embed = torch.cat(
                    [selected_embeds, selected_embeds_prime], dim=-1)
                agg_embed = torch.index_select(
                    agg_embed, -1, index=(torch.randperm(2*self.global_n_random)[:self.global_n_random]).to(agg_embed.device))

                global_random_embed[bi, pi] = agg_embed
        return global_random_embed

    def get_alignment_uniform_loss(self, x, y, x_plus_negatives, y_plus_negatives):
        '''x: (bsz,d): d is the representation dimension
           y: (bsz,d): d is the representation dimension
           x_plus_negatives: (bsz,d): d is the representation dimension. but it contain positive and negative


        '''

    def lalign(self, x, y, alpha=2):
        '''x,y: 2D . (Bs, representation dimension)
        '''
        return (x - y).norm(dim=1).pow(alpha).mean()

    def lunif(x, t=2):
        '''x: 2D. (Bs, representation dimension)
        '''
        sq_pdist = torch.pdist(x, p=2).pow(2)
        return sq_pdist.mul(-t).exp().mean().log()

    def get_global_info_nce_loss(self, global_random_embed, neg_glbal_f_embed,
                                 anchor_global_1st_ls, anchor_global_2nd_ls,
                                 n_batch):
        '''
        global_random_embed:4D
        neg_glbal_f_embed: 4D
        anchor_global_1st_ls: 3D
        anchor_global_2nd_ls: 3D


        '''
        # InfoNCE for global embedding
        total_sample = 1 + \
            global_random_embed.size(-1) + neg_glbal_f_embed.size(-1)
        # (nb, npos,embed_dim, global_n_neg+global_n_random)
        grnd_global_neg_samples = torch.cat(
            [global_random_embed, neg_glbal_f_embed], dim=-1)
        # TODO:  WHY in global_random_embed, every vector is zero vector???
        # calculate similarity among positive voxel and corresponding negative voxels
        denom_global_loss = torch.einsum(
            'bij, bijk-> bik', [anchor_global_1st_ls, grnd_global_neg_samples])/self.los_temp
        # denom_global_loss.shap -> (n_batch, self.n_pos_voxel, total_sample-1]) in other words
        # (nb, npos, total_global_neg)

        # calculate similarity of anchor positive voxel with paired positive voxel

        nom_global_loss = (torch.einsum(
            'bij, bij-> bi', [anchor_global_1st_ls, anchor_global_2nd_ls])/self.los_temp).unsqueeze(-1)
        # nom_global_loss -> (n_batch, self.n_pos_voxel,1)

        # symmetric loss, compute samethings with f'
        denom_global_loss_fi = torch.einsum(
            'bij, bijk-> bik', [anchor_global_2nd_ls, grnd_global_neg_samples])/self.los_temp
        nom_global_loss_fi = (torch.einsum(
            'bij, bij-> bi', [anchor_global_2nd_ls, anchor_global_1st_ls])/self.los_temp).unsqueeze(-1)

        # add positive and negative voxels similarity
        # (nb,npos, 1+total_global_negatives)
        denom_global_loss = torch.cat(
            [nom_global_loss, denom_global_loss], dim=-1)
        denom_global_loss_fi = torch.cat(
            [nom_global_loss_fi, denom_global_loss_fi], dim=-1)
        # print(f"Global loss detail: {denom_global_loss_fi}")

        denom_global_loss = denom_global_loss.view(
            n_batch*self.n_pos_voxel, total_sample)  # (nb*npos, 1+total_negative samples)
        denom_global_loss_fi = denom_global_loss_fi.view(
            n_batch*self.n_pos_voxel, total_sample)
        label_global = torch.zeros(denom_global_loss.shape[0], dtype=torch.long).to(
            anchor_global_2nd_ls.device)  # (n_batch*self.self.n_pos_voxel)
        # print(f"global argmax: {denom_global_loss.argmax(dim=-1)}")
        # print(f"global_fi argmax: {denom_global_loss_fi.argmax(dim=-1)}")
        '''
        # implement infonce loss
        global_loss = -torch.log(torch.exp(nom_global_loss.view(n_batch*self.n_pos_voxel))/torch.exp(denom_global_loss).sum(dim=-1))


        global_loss_fi = -torch.log(torch.exp(nom_global_loss_fi.view(n_batch*self.n_pos_voxel))/torch.exp(denom_global_loss_fi).sum(dim=-1))

        global_loss_tot = global_loss +global_loss_fi
        global_loss_tot = torch.sum(global_loss_tot)/(2*n_batch*self.n_pos_voxel)
        '''
        return denom_global_loss, denom_global_loss_fi, label_global  # global_loss_tot

    def get_local_info_nce_loss(self, anchor_local_1st_ls, anchor_local_2nd_ls, neg_local_f_embed, n_batch):

        nom_local_loss = (torch.einsum(
            'bij, bij-> bi', [anchor_local_1st_ls, anchor_local_2nd_ls])/self.los_temp).unsqueeze(-1)
        # nom_local_loss: (nb, npos,1)
        denom_local_loss = torch.einsum(
            'bij, bijk-> bik', [anchor_local_1st_ls, neg_local_f_embed])/self.los_temp

        # (nb,npos, 1+ total_local_negative)
        denom_local_loss = torch.cat(
            [nom_local_loss, denom_local_loss], dim=-1)
        # print(f"local loss detail: {denom_local_loss}")

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

        loss_label = torch.zeros(
            n_batch*self.n_pos_voxel, dtype=torch.long, device=denom_local_loss_fi.device)

        # local_loss_tot,loss_label #this   local_loss_tot,loss_label
        return denom_local_loss, denom_local_loss_fi, loss_label

    def get_local_invariance_loss(self, anchor_local_1st_ls, anchor_local_2nd_ls, n_batch):
        anchor_local_1st_ls = anchor_local_1st_ls.view(
            n_batch*self.n_pos_voxel, -1)
        anchor_local_2nd_ls = anchor_local_2nd_ls.view(
            n_batch*self.n_pos_voxel, -1)

        local_invariance = (
            anchor_local_1st_ls-anchor_local_2nd_ls).pow(2).sum(dim=-1).sqrt().mean()
        print(f"local_invariance:{local_invariance}")
        return local_invariance

    def get_global_invariance_loss(self, anchor_global_1st_ls, anchor_global_2nd_ls, n_batch):

        anchor_global_1st_ls = anchor_global_1st_ls.view(
            n_batch*self.n_pos_voxel, -1)
        anchor_global_2nd_ls = anchor_global_2nd_ls.view(
            n_batch*self.n_pos_voxel, -1)

        global_invariance = (
            anchor_global_1st_ls-anchor_global_2nd_ls).pow(2).sum(dim=-1).sqrt().mean()
        print(f"global_invariance:{global_invariance}")

        return global_invariance

    def get_dino_loss(self, anchor_global_1st_ls, anchor_global_2nd_ls,
                      anchor_local_1st_ls, anchor_local_2nd_ls,
                      n_batch):
        '''
        all tensors: (n,dim, x,y,z)

        '''
        anchor_global_1st_ls = anchor_global_1st_ls.view(
            n_batch*self.n_pos_voxel, -1)
        anchor_global_2nd_ls = anchor_global_2nd_ls.view(
            n_batch*self.n_pos_voxel, -1)

        anchor_local_1st_ls = anchor_local_1st_ls.view(
            n_batch*self.n_pos_voxel, -1)
        anchor_local_2nd_ls = anchor_local_2nd_ls.view(
            n_batch*self.n_pos_voxel, -1)

        reconst_2nd_local = self.local_lin(anchor_local_1st_ls)
        reconst_1st_local = self.local_lin(anchor_local_2nd_ls)

        reconst_2nd_global = self.global_lin(anchor_global_1st_ls)
        reconst_1st_global = self.global_lin(anchor_global_2nd_ls)

        batch_size = anchor_global_1st_ls.size(0)
        d = anchor_global_1st_ls.size(1)

        l1_mse_loss = torch.sum(
            torch.pow((anchor_local_1st_ls - reconst_1st_local), 2)) / (batch_size * d)
        l2_mse_loss = torch.sum(
            torch.pow((anchor_local_2nd_ls - reconst_2nd_local), 2)) / (batch_size * d)

        g1_mse_loss = torch.sum(
            torch.pow((anchor_global_1st_ls - reconst_1st_global), 2)) / (batch_size * d)
        g2_mse_loss = torch.sum(
            torch.pow((anchor_global_2nd_ls - reconst_2nd_global), 2)) / (batch_size * d)

        return (l1_mse_loss+l2_mse_loss+g1_mse_loss+g2_mse_loss)/4.0

    def get_local_burlow_twin_loss(self, anchor_local_1st_ls, anchor_local_2nd_ls, n_batch, device):
        anchor_local_1st_ls = (anchor_local_1st_ls - anchor_local_1st_ls.mean(
            dim=0, keepdims=True))/anchor_local_1st_ls.std(dim=0, keepdims=True)

        anchor_local_2nd_ls = (anchor_local_2nd_ls - anchor_local_2nd_ls.mean(
            dim=0, keepdims=True))/anchor_local_2nd_ls.std(dim=0, keepdims=True)

        c_local = anchor_local_1st_ls.T @ anchor_local_2nd_ls
        c_local = c_local.div_(n_batch*self.n_pos_voxel)

        print(f"diagonal local:{torch.diagonal(c_local)}")
        print(f"offdiagonal local:{torch.topk(off_diagonal(c_local),100)}")

        c_diff_local = (c_local-torch.eye(len(c_local),
                        dtype=c_local.dtype).to(c_local.device)).pow(2)
        off_diagonal(c_diff_local).mul_(0.001)
        bur_loss_local = c_diff_local.sum()

        return bur_loss_local

    def get_global_burlow_twin_loss(self, anchor_global_1st_ls, anchor_global_2nd_ls, n_batch, device):
        anchor_global_1st_ls = (anchor_global_1st_ls - anchor_global_1st_ls.mean(
            dim=0, keepdims=True))/anchor_global_1st_ls.std(dim=0, keepdims=True)

        anchor_global_2nd_ls = (anchor_global_2nd_ls - anchor_global_2nd_ls.mean(
            dim=0, keepdims=True))/anchor_global_2nd_ls.std(dim=0, keepdims=True)

        c_global = anchor_global_1st_ls.T @ anchor_global_2nd_ls

        c_global = c_global.div_(n_batch*self.n_pos_voxel)

        print(f"diagonal global:{torch.diagonal(c_global)}")
        print(f"off diagonal global:{torch.topk(off_diagonal(c_global),100)}")

        c_diff_global = (c_global-torch.eye(len(c_global),
                         dtype=c_global.dtype).to(c_global.device)).pow(2)
        off_diagonal(c_diff_global).mul_(0.001)
        bur_loss_global = c_diff_global.sum()

        return bur_loss_global

    def reconstruct_cube(self):
        pass

    def training_step(self, batch, batch_idx):
        '''
        return ([local_embedding_1st_ls ,global_embedding_1st_ls],
            [local_embedding_2nd_ls ,global_embedding_2nd_ls],[local_loss,global_loss])
        pos should be in gpu_3
        '''

        # The dataset also returns foreground masks, which only the momentum
        # variant's reconstruction loss consumes. This model has no
        # reconstruction term, so they are discarded here.
        if len(batch) == 3:
            patch_ls, pos, _unused_masks = batch
        else:
            patch_ls, pos = batch

        random_Crops, random_Crops_pair = patch_ls

        # torch.cuda.empty_cache()
        store_ls = []
        local_embedding_1st, global_embedding_1st = self.fpn(random_Crops)
        local_embedding_2nd, global_embedding_2nd = self.fpn(random_Crops_pair)

        n_batch = random_Crops.shape[0]
        W, H, D = random_Crops.shape[-3:]

        # device = random_Crops.device

        local_embedding_1st_ls = torch.nn.functional.normalize(
            local_embedding_1st, p=2, dim=1)
        global_embedding_1st_ls = torch.nn.functional.normalize(
            global_embedding_1st, p=2, dim=1)

        local_embedding_2nd_ls = torch.nn.functional.normalize(
            local_embedding_2nd, p=2, dim=1)
        global_embedding_2nd_ls = torch.nn.functional.normalize(
            global_embedding_2nd, p=2, dim=1)

        # shape:  local_embedding_1st_ls,local_embedding_2nd_ls : shape (nb, embed_dim,patch_shape[0]//2,patch_shape[1]//2,patch_shape[2]//2)
        #        global_embedding_1st_ls, global_embedding_2nd_ls : shape (nb, embed_dim,patch_shape[0]//16,patch_shape[1]//16,patch_shape[2]//16)
        #                                                          or shape (nb, embed_dim,patch_shape[0]//16,patch_shape[1]//16,patch_shape[2]//4)

        # make global embedding as same size of local embedding
        b, c, w, h, d = local_embedding_2nd_ls.shape  # x,y,z

        # make the coordinates of voxels of one anchor pair in the same axis.
        # shape (nb, n_pos_voxel,6)
        pos = pos.view(n_batch, self.n_pos_voxel, -1).to(dtype=torch.long)

        # get the anchor voxels embedding
        # anchor_global_1st_ls : (nb, n_pos_voxel, embedding_dim)
        (anchor_global_1st_ls, anchor_global_2nd_ls,
         anchor_local_1st_ls, anchor_local_2nd_ls), (pos_z) = self.get_local_global_anchors(local_embedding_1st_ls, local_embedding_2nd_ls,
                                                                                            global_embedding_1st_ls, global_embedding_2nd_ls, pos, n_batch)

        # get the similarity tensors of all voxels with anchor voxels
        # print(f"Similarity Matrix calculation started")
        (sg, sg_prime, sl, sl_prime) = self.get_similarity_tensors(anchor_global_1st_ls,
                                                                   global_embedding_1st_ls, global_embedding_2nd_ls,
                                                                   anchor_local_1st_ls,
                                                                   local_embedding_1st_ls, local_embedding_2nd_ls)
        # print(f"Similarity Matrix calculation Ended")
        sl_x, sl_z, sl_y = sl.size()[-3:]
        sg_local = sg.clone()
        sg_prime_local = sg_prime.clone()
        # (nb, n_pos_voxel, patche_shape//2)
        sg_local = F.upsample(sg_local, size=(
            sl_x, sl_z, sl_y), mode='trilinear')

        sg_prime_local = F.upsample(sg_prime_local, size=(
            sl_x, sl_z, sl_y), mode='trilinear')  # (nb, n_pos_voxel, patche_shape//2)
        # get the negative global voxels embedding
        # neg_glbal_f_embed (n_batch,self.n_pos_voxel,self.embedding_dim,self.global_n_neg)
        # print(f"get_global_negative_voxels started")
        neg_glbal_f_embed = self.get_global_negative_voxels(sg, sg_prime, pos_z,
                                                            global_embedding_1st_ls, global_embedding_2nd_ls,
                                                            n_batch, b)
        # print(f"get_global_negative_voxels Ended")
        # get local negative voxel embedding
        # neg_local_f_embed :(n_batch,self.n_pos_voxel,self.embedding_dim,self.local_cand_neg)
        # print(f"get_hard_local_negative_voxels started")
        neg_local_f_embed = self.get_hard_local_negative_voxels(sl, sl_prime,
                                                                sg_local, sg_prime_local,
                                                                local_embedding_1st_ls, local_embedding_2nd_ls,
                                                                n_batch, pos, W, H, D, b)
        # print(f"get_hard_local_negative_voxels Ended")
        # get across-batch global negative voxel embeddings
        dtype_ = neg_local_f_embed.dtype
        # global_random_embed : (n_batch,self.n_pos_voxel,self.embedding_dim,self.global_n_random)
        # print(f"get_across_global_negative_voxels Started")
        global_random_embed = self.get_across_global_negative_voxels(global_embedding_1st_ls,
                                                                     global_embedding_2nd_ls,
                                                                     n_batch, dtype_)
        # print(f"get_across_global_negative_voxels Ened")
        # --- Get Info-NCE Loss---
        # get global info_nce_loss
        # print(f"get_global_info_nce_loss Started")
        denom_global_loss, denom_global_loss_fi, label_global = self. get_global_info_nce_loss(global_random_embed, neg_glbal_f_embed,
                                                                                               anchor_global_1st_ls, anchor_global_2nd_ls,
                                                                                               n_batch)
        # print(f"get_global_info_nce_loss Ended")
        # get local info_nce_loss
        # previously it was local_loss_tot,loss_label
        # print(f"get_local_info_nce_loss Started")
        denom_local_loss, denom_local_loss_fi, loss_label = self.get_local_info_nce_loss(anchor_local_1st_ls,
                                                                                         anchor_local_2nd_ls,
                                                                                         neg_local_f_embed, n_batch)
        # print(f"get_local_info_nce_loss Ended")

        # del neg_local_f_embed
        # del global_random_embed
        # del neg_glbal_f_embed
        # del grnd_global_neg_samples
        local_1, global_1, local_2, global_2 = self.get_instance_embedding(local_embedding_1st, global_embedding_1st,
                                                                           local_embedding_2nd, global_embedding_2nd,
                                                                           )
        instance_loss_1 = self.info_nce_instance(local_1, global_1, local_2, global_2,
                                                 temperature=self.model_kwargs['los_temp_instance'], global_loss_wt=0.5)
        instance_loss_2 = self.info_nce_instance(local_2, global_2, local_1, global_1,
                                                 temperature=self.model_kwargs['los_temp_instance'], global_loss_wt=0.5)
        instance_loss = (instance_loss_1 + instance_loss_2)/2

        local_loss_1 = torch.nn.CrossEntropyLoss()(denom_local_loss, loss_label)
        local_loss_2 = torch.nn.CrossEntropyLoss()(denom_local_loss_fi, loss_label)
        local_loss_tot = (local_loss_1 + local_loss_2)/2

        global_loss_1 = torch.nn.CrossEntropyLoss()(denom_global_loss, label_global)
        global_loss_2 = torch.nn.CrossEntropyLoss()(denom_global_loss_fi, label_global)
        global_loss_tot = (global_loss_1 + global_loss_2)/2

        loss = instance_loss + global_loss_tot + local_loss_tot

        self.log('train_local_loss', local_loss_tot, on_step=True,
                 prog_bar=True, sync_dist=True, logger=True)
        self.log('train_global_loss', global_loss_tot, on_step=True,
                 prog_bar=True, sync_dist=True, logger=True)
        self.log('train_instance_loss', instance_loss, on_step=True,
                 prog_bar=True, sync_dist=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        '''
        return ([local_embedding_1st_ls ,global_embedding_1st_ls],
            [local_embedding_2nd_ls ,global_embedding_2nd_ls],[local_loss,global_loss])
        pos should be in gpu_3
        '''

        # The dataset also returns foreground masks, which only the momentum
        # variant's reconstruction loss consumes. This model has no
        # reconstruction term, so they are discarded here.
        if len(batch) == 3:
            patch_ls, pos, _unused_masks = batch
        else:
            patch_ls, pos = batch

        random_Crops, random_Crops_pair = patch_ls
        # torch.cuda.empty_cache()
        store_ls = []
        local_embedding_1st, global_embedding_1st = self.fpn(random_Crops)
        local_embedding_2nd, global_embedding_2nd = self.fpn(random_Crops_pair)

        n_batch = random_Crops.shape[0]
        W, H, D = random_Crops.shape[-3:]

        # device = random_Crops.device

        local_embedding_1st_ls = torch.nn.functional.normalize(
            local_embedding_1st, p=2, dim=1)
        global_embedding_1st_ls = torch.nn.functional.normalize(
            global_embedding_1st, p=2, dim=1)

        local_embedding_2nd_ls = torch.nn.functional.normalize(
            local_embedding_2nd, p=2, dim=1)
        global_embedding_2nd_ls = torch.nn.functional.normalize(
            global_embedding_2nd, p=2, dim=1)

        # shape:  local_embedding_1st_ls,local_embedding_2nd_ls : shape (nb, embed_dim,patch_shape[0]//2,patch_shape[1]//2,patch_shape[2]//2)
        #        global_embedding_1st_ls, global_embedding_2nd_ls : shape (nb, embed_dim,patch_shape[0]//16,patch_shape[1]//16,patch_shape[2]//16)
        #                                                          or shape (nb, embed_dim,patch_shape[0]//16,patch_shape[1]//16,patch_shape[2]//4)

        # make global embedding as same size of local embedding
        b, c, w, h, d = local_embedding_2nd_ls.shape  # x,y,z

        # make the coordinates of voxels of one anchor pair in the same axis.
        # shape (nb, n_pos_voxel,6)
        pos = pos.view(n_batch, self.n_pos_voxel, -1).to(dtype=torch.long)

        # get the anchor voxels embedding
        # anchor_global_1st_ls : (nb, n_pos_voxel, embedding_dim)
        (anchor_global_1st_ls, anchor_global_2nd_ls,
         anchor_local_1st_ls, anchor_local_2nd_ls), (pos_z) = self.get_local_global_anchors(local_embedding_1st_ls, local_embedding_2nd_ls,
                                                                                            global_embedding_1st_ls, global_embedding_2nd_ls, pos, n_batch)

        # get the similarity tensors of all voxels with anchor voxels
        # print(f"Similarity Matrix calculation started")
        (sg, sg_prime, sl, sl_prime) = self.get_similarity_tensors(anchor_global_1st_ls,
                                                                   global_embedding_1st_ls, global_embedding_2nd_ls,
                                                                   anchor_local_1st_ls,
                                                                   local_embedding_1st_ls, local_embedding_2nd_ls)
        # print(f"Similarity Matrix calculation Ended")
        sl_x, sl_z, sl_y = sl.size()[-3:]
        sg_local = sg.clone()
        sg_prime_local = sg_prime.clone()
        # (nb, n_pos_voxel, patche_shape//2)
        sg_local = F.upsample(sg_local, size=(
            sl_x, sl_z, sl_y), mode='trilinear')

        sg_prime_local = F.upsample(sg_prime_local, size=(
            sl_x, sl_z, sl_y), mode='trilinear')  # (nb, n_pos_voxel, patche_shape//2)
        # get the negative global voxels embedding
        # neg_glbal_f_embed (n_batch,self.n_pos_voxel,self.embedding_dim,self.global_n_neg)
        # print(f"get_global_negative_voxels started")
        neg_glbal_f_embed = self.get_global_negative_voxels(sg, sg_prime, pos_z,
                                                            global_embedding_1st_ls, global_embedding_2nd_ls,
                                                            n_batch, b)
        # print(f"get_global_negative_voxels Ended")
        # get local negative voxel embedding
        # neg_local_f_embed :(n_batch,self.n_pos_voxel,self.embedding_dim,self.local_cand_neg)
        # print(f"get_hard_local_negative_voxels started")
        neg_local_f_embed = self.get_hard_local_negative_voxels(sl, sl_prime,
                                                                sg_local, sg_prime_local,
                                                                local_embedding_1st_ls, local_embedding_2nd_ls,
                                                                n_batch, pos, W, H, D, b)
        # print(f"get_hard_local_negative_voxels Ended")
        # get across-batch global negative voxel embeddings
        dtype_ = neg_local_f_embed.dtype
        # global_random_embed : (n_batch,self.n_pos_voxel,self.embedding_dim,self.global_n_random)
        # print(f"get_across_global_negative_voxels Started")
        global_random_embed = self.get_across_global_negative_voxels(global_embedding_1st_ls,
                                                                     global_embedding_2nd_ls,
                                                                     n_batch, dtype_)
        # print(f"get_across_global_negative_voxels Ened")
        # --- Get Info-NCE Loss---
        # get global info_nce_loss
        # print(f"get_global_info_nce_loss Started")
        denom_global_loss, denom_global_loss_fi, label_global = self. get_global_info_nce_loss(global_random_embed, neg_glbal_f_embed,
                                                                                               anchor_global_1st_ls, anchor_global_2nd_ls,
                                                                                               n_batch)
        # print(f"get_global_info_nce_loss Ended")
        # get local info_nce_loss
        # previously it was local_loss_tot,loss_label
        # print(f"get_local_info_nce_loss Started")
        denom_local_loss, denom_local_loss_fi, loss_label = self.get_local_info_nce_loss(anchor_local_1st_ls,
                                                                                         anchor_local_2nd_ls,
                                                                                         neg_local_f_embed, n_batch)
        # print(f"get_local_info_nce_loss Ended")

        # del neg_local_f_embed
        # del global_random_embed
        # del neg_glbal_f_embed
        # del grnd_global_neg_samples
        local_1, global_1, local_2, global_2 = self.get_instance_embedding(local_embedding_1st, global_embedding_1st,
                                                                           local_embedding_2nd, global_embedding_2nd,
                                                                           )

        instance_loss_1 = self.info_nce_instance(local_1, global_1, local_2, global_2,
                                                 temperature=self.model_kwargs['los_temp_instance'], global_loss_wt=0.5)
        instance_loss_2 = self.info_nce_instance(local_2, global_2, local_1, global_1,
                                                 temperature=self.model_kwargs['los_temp_instance'], global_loss_wt=0.5)
        instance_loss = (instance_loss_1 + instance_loss_2)/2

        local_loss_1 = torch.nn.CrossEntropyLoss()(denom_local_loss, loss_label)
        local_loss_2 = torch.nn.CrossEntropyLoss()(denom_local_loss_fi, loss_label)
        local_loss_tot = (local_loss_1 + local_loss_2)/2

        global_loss_1 = torch.nn.CrossEntropyLoss()(denom_global_loss, label_global)
        global_loss_2 = torch.nn.CrossEntropyLoss()(denom_global_loss_fi, label_global)
        global_loss_tot = (global_loss_1 + global_loss_2)/2

        loss = instance_loss + global_loss_tot + local_loss_tot

        self.log('val_local_loss', local_loss_tot, on_step=True,
                 prog_bar=True, sync_dist=True, logger=True)
        self.log('val_global_loss', global_loss_tot, on_step=True,
                 prog_bar=True, sync_dist=True, logger=True)
        self.log('val_instance_loss', instance_loss, on_step=True,
                 prog_bar=True, sync_dist=True, logger=True)

        return loss

    def configure_optimizers(self):

        print("Configuring optimizers...")
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=float(self.lr),
            weight_decay=float(self.weight_decay)
        )
        train_dataloader = self.trainer.datamodule.train_dataloader()
        if train_dataloader is None:
            raise ValueError(
                "train_dataloader is None. Ensure your datamodule is correctly implemented.")

        print(f"Total steps per epoch: {len(train_dataloader)}")

        epochs = self.trainer.max_epochs
        if epochs is None:
            raise ValueError(
                "self.trainer.max_epochs is None. Ensure you have set max_epochs in your Trainer.")

        lr_scheduler_config = {
            'scheduler': OneCycleLR(
                optimizer,
                max_lr=self.lr,
                steps_per_epoch=len(train_dataloader),
                epochs=epochs,
                pct_start=float(self.model_kwargs['pct_start']),
                div_factor=float(self.model_kwargs['div_factor']),
                final_div_factor=int(self.model_kwargs['final_div_factor'])



            ),
            "interval": "step",
            "name": 'oneclr'
        }
        print("Optimizer and scheduler configured successfully.")
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


class Model_Momentum(Model):

    def __init__(self, model_kwargs):
        super(Model_Momentum, self).__init__(model_kwargs)
        self.model_kwargs = model_kwargs
        self.apply_z_stride = self.model_kwargs['apply_z_stride'] if 'apply_z_stride' in self.model_kwargs else [
            2, 1, 1, 2]

        self.n_batch = self.model_kwargs['batch_size']
        self.n_pos_voxel = self.model_kwargs['n_pos_voxel']
        self.patch_size = self.model_kwargs['patch_size']

        self.embedding_dim = self.model_kwargs['embedding_dim']
        self.lr = self.model_kwargs['lr']
        self.los_temp = self.model_kwargs['los_temp']

        self.weight_decay = self.model_kwargs['weight_decay']
#        self.device = self.model_kwargs['device'] if 'device' in self.model_kwargs else torch.device(
#            'cpu')
        self.radius = self.model_kwargs['radius'] if 'radius' in self.model_kwargs else 1
        if sum(self.model_kwargs['apply_z_stride']) == 6:
            self.model_kwargs['equal_stride'] = False
        else:
            self.model_kwargs['equal_stride'] = True
        print(f"apply_z_stride:{self.apply_z_stride}")
        print(f"equal_stride:{self.model_kwargs['equal_stride']}")

        self.factor_n_negative_samples = self.model_kwargs[
            'factor_n_negative_samples'] if 'factor_n_negative_samples' in self.model_kwargs else 40
        print(f"factor_n_negative_samples:{self.factor_n_negative_samples}")
        # add mlp projection head

        self.scaling_factor = (
            self.patch_size[0]*self.patch_size[1]*self.patch_size[2])/(96*96*96)

        if reduce(mul, self.apply_z_stride, 1) == 16:
            self.global_n_neg = 190  #
        else:
            # valid for [2,1,1,2] z-stride
            self.global_n_neg = int(500*(self.scaling_factor))

        self.local_cand_neg = int(20000*(self.scaling_factor))

        self.global_n_random = self.global_n_neg*(self.n_batch-1)
        self.momentum = model_kwargs['momentum']
        self.save_hyperparameters()
        print(f"Patch Size:{self.patch_size}")
        print(f"Batch Size:{self.n_batch}")
        print(f"scaling_factor:{self.scaling_factor}")
        print(f"n_pos_voxel:{self.n_pos_voxel}")
        print(f"global_n_neg:{self.global_n_neg}")
        print(f"embedding_dim:{self.embedding_dim}")
        print(f"local_cand_neg:{self.local_cand_neg}")
        print(
            f"local_neg:{self.local_cand_neg//self.model_kwargs['factor_n_negative_samples']}")
        print(f"global_n_random:{self.global_n_random}")
        print(f"lr: {self.lr}")
        self.reconstruction_loss = self.model_kwargs['reconstruction_loss']
        if self.reconstruction_loss:
            print('Reconstruction loss is enabled')
            self.decoderlayer = self.decoder_block(self.embedding_dim, 16)
            self.transconv = self.conv_transpose(
                16, input_padding=(0, 0, 0))
            self.last_cnn = self.last_CNN_block(16, 1)

        self.fpn = FPN(apply_z_stride=self.apply_z_stride)
        self.fpn_k = FPN(apply_z_stride=self.apply_z_stride)
        for param_q, param_k in zip(self.fpn.parameters(), self.fpn_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.instance_project_head_local = Proj_Head(
            in_dim=128, inner_dim=128, out_dim=128)
        self.instance_project_head_local_k = Proj_Head(
            in_dim=128, inner_dim=128, out_dim=128)
        # self.instance_pred_head_local = Pred_Head( in_dim=128, inner_dim=4096, out_dim=128 )

        self.instance_project_head_global = Proj_Head(
            in_dim=128, inner_dim=128, out_dim=128)
        self.instance_project_head_global_k = Proj_Head(
            in_dim=128, inner_dim=128, out_dim=128)

        for param_q, param_k in zip(self.instance_project_head_local.parameters(), self.instance_project_head_local_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        for param_q, param_k in zip(self.instance_project_head_global.parameters(), self.instance_project_head_global_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # self.instance_pred_head_global = Pred_Head( in_dim=128, inner_dim=4096, out_dim=128 )
        self.avgpool_local = nn.AvgPool3d(48, stride=1)
        self.avgpool_global = nn.AvgPool3d(6, stride=1)

        # self.projection = nn.Sequential(
        #     nn.Linear(in_features=128, out_features=5*128),
        #     nn.BatchNorm1d(5*128),
        #     nn.ReLU(),
        #     nn.Linear(in_features=5*128, out_features=128),
        #     nn.BatchNorm1d(128),
        # )

        self.K = None
        self.k = None

        self.memory_bank_size = self.model_kwargs['memory_bank_size']
        self.embedding_dim = self.model_kwargs['embedding_dim']

        self.register_buffer('memory_bank_local', torch.randn(
            self.memory_bank_size, self.embedding_dim))
        self.register_buffer(
            'memory_pointer_local', torch.zeros(1, dtype=torch.long))

        self.register_buffer('memory_bank_global', torch.randn(
            self.memory_bank_size, self.embedding_dim))
        self.register_buffer(
            'memory_pointer_global', torch.zeros(1, dtype=torch.long))

    def conv_transpose(self, output_channels, input_padding):
        conv_t = nn.ConvTranspose3d(
            output_channels,
            output_channels,
            kernel_size=2,
            stride=2,
            padding=input_padding,
        )

        return conv_t

    def decoder_block(self, input_channels, output_channels, input_padding=(0, 0, 0)):
        normalizer = 'batchnorm'
        decoder = nn.Sequential(
            nn.Conv3d(input_channels, output_channels,
                      kernel_size=3, padding=1,),
            nn.BatchNorm3d(output_channels) if normalizer == 'batchnorm' else torch.nn.GroupNorm(
                self.num_group, output_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(output_channels, output_channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm3d(output_channels) if normalizer == 'batchnorm' else torch.nn.GroupNorm(
                self.num_group, output_channels),
            nn.LeakyReLU(inplace=True),
        )
        return decoder

    def last_CNN_block(self, input_channels, output_channels, padding=1):
        normalizer = 'batchnorm'
        cnn_block = nn.Sequential(
            nn.Conv3d(input_channels, input_channels,
                      kernel_size=3, padding=padding),

            nn.BatchNorm3d(input_channels) if normalizer == 'batchnorm' else torch.nn.GroupNorm(
                self.num_group, input_channels),

            nn.LeakyReLU(inplace=True),

            nn.Conv3d(input_channels, input_channels,
                      kernel_size=3, padding=padding),

            nn.BatchNorm3d(input_channels) if normalizer == 'batchnorm' else torch.nn.GroupNorm(
                self.num_group, input_channels),

            nn.LeakyReLU(inplace=True),

            nn.Conv3d(input_channels, output_channels, kernel_size=1),
        )
        return cnn_block

    def setup(self, stage=None):
        world_size = self.trainer.world_size if self.trainer else 1
        self.K = int(self.model_kwargs['num_instances'] * 1.0 / world_size /
                     self.model_kwargs['batch_size'] * self.trainer.max_epochs)
        self.k = int(self.model_kwargs['num_instances'] * 1.0 / world_size /
                     self.model_kwargs['batch_size'] * self.trainer.current_epoch)

    def on_train_start(self):
        world_size = self.trainer.world_size if self.trainer else 1
        self.K = int(self.model_kwargs['num_instances'] * 1.0 / world_size /
                     self.model_kwargs['batch_size'] * self.trainer.max_epochs)
        self.k = int(self.model_kwargs['num_instances'] * 1.0 / world_size /
                     self.model_kwargs['batch_size'] * (self.trainer.current_epoch))

    def _momentum_update_key_encoder(self):

        _contrast_momentum = 1. - \
            (1-self.momentum) * \
            (np.cos(np.pi*self.k/self.K)+1)/2.

        for param_q, param_k in zip(self.fpn.parameters(), self.fpn_k.parameters()):
            param_k.data = param_k.data*_contrast_momentum + \
                param_q.data*(1. - _contrast_momentum)

        for param_q, param_k in zip(self.instance_project_head_local.parameters(), self.instance_project_head_local_k.parameters()):
            param_k.data = param_k.data*_contrast_momentum + \
                param_q.data*(1. - _contrast_momentum)

        for param_q, param_k in zip(self.instance_project_head_global.parameters(), self.instance_project_head_global_k.parameters()):
            param_k.data = param_k.data*_contrast_momentum + \
                param_q.data*(1. - _contrast_momentum)

    def get_similarity_tensors(self, anchor_global_1st_ls,
                               global_embedding_1st_ls_k, global_embedding_2nd_ls_k,
                               anchor_local_1st_ls,
                               local_embedding_1st_ls_k, local_embedding_2nd_ls_k
                               ):

        # step 7: calculate similarity maps sg, sg',sl,sl'

        # sg would be the similarity map of anchor_global_1st_ls with global_embedding_1st_ls
        sg = (torch.einsum('bij, bjklm-> biklm',
              [anchor_global_1st_ls, global_embedding_1st_ls_k])/torch.tensor(128).sqrt()).to(anchor_global_1st_ls.device)
        # sg' would be the similarity map of anchor_global_1st_ls with global_embedding_2nd_ls
        sg_prime = (torch.einsum('bij, bjklm-> biklm',
                    [anchor_global_1st_ls, global_embedding_2nd_ls_k])/torch.tensor(128).sqrt()).to(anchor_global_1st_ls.device)
        # shape : sg,sg_prime:
        # (nb, n_pos_voxel, patche_shape//16) or (nb, n_pos_voxel, patche_shape[0]//16, patche_shape[0]//16,patche_shape[0]//4) depend of apply_z_stride

        # lg would be the similarity map of anchor_local_1st_ls with local_embedding_1st_ls
        sl = (torch.einsum('bij, bjklm-> biklm',
              [anchor_local_1st_ls, local_embedding_1st_ls_k])/torch.tensor(128).sqrt()).to(anchor_local_1st_ls.device)
        # sg' would be the similarity map of anchor_global_1st_ls with global_embedding_2nd_ls
        sl_prime = (torch.einsum('bij, bjklm-> biklm',
                    [anchor_local_1st_ls, local_embedding_2nd_ls_k])/torch.tensor(128).sqrt()).to(anchor_local_1st_ls.device)
        # shape: sl,sl_prime: (nb, n_pos_voxel, patche_shape//2)

        return (sg, sg_prime, sl, sl_prime)

    def get_global_negative_voxels(self, sg, sg_prime, pos_z, global_embedding_1st_ls_k,
                                   global_embedding_2nd_ls_k, n_batch, b):

        # set similarity of positive voxel with themselves to zero

        sg[torch.arange(n_batch).reshape(n_batch, -1).long().to(sg.device), torch.arange(self.n_pos_voxel).long().to(sg.device), pos_z[torch.arange(n_batch).long().to(sg.device),
                                                                                                                                       :, 0], pos_z[torch.arange(n_batch).long().to(sg.device), :, 1], pos_z[torch.arange(n_batch).long().to(sg.device), :, 2]] = torch.tensor(-float("inf")).to(sg.device)

        sg_prime[torch.arange(n_batch).reshape(n_batch, -1).long().to(sg_prime.device), torch.arange(self.n_pos_voxel).long().to(sg_prime.device), pos_z[torch.arange(n_batch).long().to(
            sg_prime.device), :, 3], pos_z[torch.arange(n_batch).long().to(sg_prime.device), :, 4], pos_z[torch.arange(n_batch).long().to(sg_prime.device), :, 5]] = torch.tensor(-float("inf")).to(sg_prime.device)

        # shape: sg,sg_prime
        # (nb, n_pos_voxel, patche_shape//16) or #
        # (nb, n_pos_voxel, patche_shape[0]//16, patche_shape[0]//16,patche_shape[0]//4) depend of apply_z_stride

        # sort the sg,sg_prime, sl,sl_prime
        # (nb, n_pos_voxel, product of (patch_shape//16))
        sg = sg.view(b, self.n_pos_voxel, -1)
        # (nb, n_pos_voxel, product of (patch_shape//16))
        sg_prime = sg_prime.view(b, self.n_pos_voxel, -1)
        _, indices_g = torch.sort(sg, descending=True)
        # (nb, n_pos_voxel, product of (patch_shape//16)), But last time,checked not.
        # TODO. we will log the indices_g, indices_g_prime with coordinates of positive voxels to see which
        # distribution of distace of negative samples coordinate from the positive samples coordinate
        _, indices_g_prime = torch.sort(sg_prime, descending=True)

        # (nb, n_pos_voxel, self.global_n_neg)
        indices_g = indices_g[..., :self.global_n_neg]
        # (nb, n_pos_voxel, self.global_n_neg)
        indices_g_prime = indices_g_prime[..., :self.global_n_neg]
        # indices_g.shape -> torch.Size([n_batch, self.self.n_pos_voxel, global_n_neg])
        # store_ls.append(indices_g.clone().cpu())
        # store_ls.append(indices_g_prime.clone().cpu())

        global_embedding_1st_ls_flat = global_embedding_1st_ls_k.reshape(
            n_batch, self.embedding_dim, -1)
        global_embedding_2nd_ls_flat = global_embedding_2nd_ls_k.reshape(
            n_batch, self.embedding_dim, -1)
        # shape: global_embedding_1st_ls_flat,global_embedding_2nd_ls_flat
        # (nb, embedding dim,  product of patche_shape//16)

        # global_embedding_1st_ls_flat.shape -> (n_batch,self.self.n_pos_voxel,self.self.embedding_dim,dim_x,dim_z*dim_y)

        global_embedding_1st_ls_flat = global_embedding_1st_ls_flat.unsqueeze(
            1).expand(-1, self.n_pos_voxel, -1, -1)
        global_embedding_2nd_ls_flat = global_embedding_2nd_ls_flat.unsqueeze(
            1).expand(-1, self.n_pos_voxel, -1, -1)
        # shape: global_embedding_1st_ls_flat,global_embedding_2nd_ls_flat
        # (nb, n_pos_voxel, embedding_dim,product of patche_shape//16)

        # our target global hard negative sample should be [3,4,128,10]
        indices_g = indices_g.unsqueeze(
            2).expand(-1, -1, self.embedding_dim, -1)
        # (n_batch,self.n_pos_voxel,self.embedding_dim,self.global_n_neg)
        indices_g_prime = indices_g_prime.unsqueeze(
            2).expand(-1, -1, self.embedding_dim, -1)
        # indices_g,indices_g_prime
        # (nb,n_pos_voxel, embedding dim, global_n_neg)

        neg_global_emb_g = torch.gather(
            global_embedding_1st_ls_flat, dim=-1, index=indices_g)
        neg_global_emb_g_prime = torch.gather(
            global_embedding_2nd_ls_flat, dim=-1, index=indices_g_prime)
        # neg_global_emb_g,neg_global_emb_g_prime
        # (nb,n_pos_voxel, embedding dim, global_n_neg)

        # del global_embedding_1st_ls_flat
        # del global_embedding_2nd_ls_flat
        # neg_global_emb_g_prime -> (n_batch,self.n_pos_voxel,self.self.embedding_dim,self.global_n_neg)

        # concat global hard embedding from sg,sg_prime and randomly choose self_global_n_neg
        # (nb,n_pos_voxel, embedding dim, 2*global_n_neg)
        neg_global_emb = torch.cat(
            [neg_global_emb_g, neg_global_emb_g_prime], dim=-1)
        # neg_global_emb_g,neg_global_emb_g_prime
        # del neg_global_emb_g
        # del neg_global_emb_g_prime
        # neg_global_emb.shape -> ((n_batch,self.n_pos_voxel,self.self.embedding_dim,2*self.global_n_neg))
        neg_glbal_f_embed = torch.index_select(neg_global_emb, dim=-1, index=(
            torch.randperm(2*self.global_n_neg)[:self.global_n_neg]).to(neg_global_emb.device))
        # shape : neg_glbal_f_embed:(n_batch,self.n_pos_voxel,self.embedding_dim,self.global_n_neg)

        return neg_glbal_f_embed

    def get_hard_local_negative_voxels(self, sl, sl_prime, sg_local, sg_prime_local,
                                       local_embedding_1st_ls_k, local_embedding_2nd_ls_k,
                                       n_batch, pos, W, H, D, b):

        sl = sl+sg_local  # (nb,n_pos_voxel, patch_shap//2)
        sl_prime = sl_prime + sg_prime_local

        # instead of masking only the positive voxel, mask all the voxels in the radius

        (left_pos_x, left_pos_z, left_pos_y), (right_pos_x, right_pos_z,
                                               right_pos_y) = self.get_voxel_coordinates_for_local_negative_voxel_selection(pos, W, H, D)
        for ib in range(n_batch):
            for ip in range(self.n_pos_voxel):
                # sl

                sl[ib, ip, left_pos_x[ib, ip, 0].item():right_pos_x[ib, ip, 0].item()+1,
                   left_pos_y[ib, ip, 0].item():right_pos_y[ib, ip, 0].item()+1,
                   left_pos_z[ib, ip, 0].item():right_pos_z[ib, ip, 0].item()+1] = torch.tensor(-float("inf"))

                # sl_prime
                sl_prime[ib, ip, left_pos_x[ib, ip, 1].item():right_pos_x[ib, ip, 1].item()+1,
                         left_pos_y[ib, ip, 1].item():right_pos_y[ib, ip, 1].item()+1,
                         left_pos_z[ib, ip, 1].item():right_pos_z[ib, ip, 1].item()+1] = torch.tensor(-float("inf"))

        # TODO: if radius =1, then there is no circle, only the positive voxel's similarity score is -inf
        # try with larger radius like 3, done in config file
        # (nb,n_pos_voxels, product of patch shape//2)
        sl = sl.view(b, self.n_pos_voxel, -1)
        sl_prime = sl_prime.view(b, self.n_pos_voxel, -1)
        # shape: sl,sl_prime (nb,n_pos_voxel, product of (patch_shape//2))

        # shape: indices_sgl,indices_sgl (nb,n_pos_voxel, product of (patch_shape//2))
        _, indices_sgl = torch.sort(sl, descending=True)
        _, indices_sgl_prime = torch.sort(sl_prime, descending=True)

        # del sl
        # del sl_prime

        # del sg
        # del sg_prime

        # shape: indices_sgl,indices_sgl (nb,n_pos_voxel, local_cand_neg)
        indices_sgl = indices_sgl[..., :self.local_cand_neg //
                                  self.factor_n_negative_samples]
        # shape: indices_sgl,indices_sgl (nb,n_pos_voxel, local_cand_neg)
        indices_sgl_prime = indices_sgl_prime[...,
                                              :self.local_cand_neg//self.factor_n_negative_samples]
        # store_ls.append(indices_sgl.clone().cpu())
        # store_ls.append(indices_sgl_prime.clone().cpu())
        # indices_sgl.shape -> (n_batch,self.self.n_pos_voxel,self.local_cand_neg)
        local_embedding_1st_ls_flat = local_embedding_1st_ls_k.reshape(
            b, self.embedding_dim, -1)
        local_embedding_2nd_ls_flat = local_embedding_2nd_ls_k.reshape(
            b, self.embedding_dim, -1)
        # shape:local_embedding_*_ls_flat: (nb, n_pos_voxel, product of patch_shape//2)
        local_embedding_1st_ls_flat = local_embedding_1st_ls_flat.unsqueeze(
            1).expand(-1, self.n_pos_voxel, -1, -1)
        local_embedding_2nd_ls_flat = local_embedding_2nd_ls_flat.unsqueeze(
            1).expand(-1, self.n_pos_voxel, -1, -1)

        # shape: local_embedding_*_ls_flat.shape -> (n_batch,self.n_pos_voxel,self.embedding_dim,product of patch_shape//2)

        # (nb,n_pos_voxel, embedding_dim, local_cand_neg)
        indices_sgl = indices_sgl.unsqueeze(
            2).expand(-1, -1, self.embedding_dim, -1)
        indices_sgl_prime = indices_sgl_prime.unsqueeze(
            2).expand(-1, -1, self.embedding_dim, -1)
        # indices_sgl_prime -> (n_batch,self.n_pos_voxel,self.self.embedding_dim,self.local_cand_neg)

        # (nb,n_pos_voxel, embedding_dim, local_cand_neg)
        neg_local_emb_l = torch.gather(
            local_embedding_1st_ls_flat, dim=-1, index=indices_sgl)
        neg_local_emb_l_prime = torch.gather(
            local_embedding_2nd_ls_flat, dim=-1, index=indices_sgl_prime)
        # del local_embedding_2nd_ls_flat
        # del local_embedding_1st_ls_flat
        # concate local_hard_negative from sl,sl_prime
        # shape :neg_local_emb_l,neg_local_emb_l_prime (n_batch,self.n_pos_voxel,self.local_cand_neg)
        neg_local_emb = torch.cat(
            [neg_local_emb_l, neg_local_emb_l_prime], dim=-1)
        # shape: neg_local_emb :(n_batch,self.n_pos_voxel,self.embedding_dim,2*self.local_cand_neg)

        # del neg_local_emb_l
        # del neg_local_emb_l_prime

        neg_local_f_embed = torch.index_select(neg_local_emb, dim=-1, index=(torch.randperm(
            2*(self.local_cand_neg//self.factor_n_negative_samples))[:(self.local_cand_neg//self.factor_n_negative_samples)]).to(neg_local_emb.device))

        return neg_local_f_embed

    def get_across_global_negative_voxels(self, global_embedding_1st_ls_k,
                                          global_embedding_2nd_ls_k,
                                          n_batch, dtype_):

        global_random_embed = torch.empty(
            (n_batch, self.n_pos_voxel, self.embedding_dim, self.global_n_random), dtype=dtype_, device=global_embedding_1st_ls_k.device)
        for bi in range(n_batch):
            for pi in range(self.n_pos_voxel):
                # excluding the sample of current positive voxel
                candidate_samples = torch.cat(
                    [global_embedding_1st_ls_k[0:bi], global_embedding_1st_ls_k[bi+1:]], dim=0)
                # shape: candidate_samples (n_batch-1,self.n_pos_voxel, patch_shape//16 ) if appy_equal_stried ==True
                candidate_samples = candidate_samples.permute(1, 0, 2, 3, 4).reshape(
                    self.embedding_dim, -1)  # (embedding_dim, n_batch*product of patch//16)
                selected_embeds = torch.index_select(candidate_samples, dim=-1, index=(torch.randperm(
                    candidate_samples.shape[-1])[:self.global_n_random]).to(global_embedding_1st_ls_k.device))  # (embedding_dim, global_n_random)

                # global_prime
                candidate_samples_prime = torch.cat(
                    [global_embedding_2nd_ls_k[0:bi], global_embedding_2nd_ls_k[bi+1:]], dim=0)
                # shape: candidate_samples (n_batch-1,self.n_pos_voxel, patch_shape//16 ) if appy_equal_stried ==True

                candidate_samples_prime = candidate_samples_prime.permute(
                    1, 0, 2, 3, 4).reshape(self.embedding_dim, -1).to(global_embedding_2nd_ls_k.device)
                selected_embeds_prime = torch.index_select(candidate_samples_prime, dim=-1, index=(
                    torch.randperm(candidate_samples_prime.shape[-1])[:self.global_n_random]).to(global_embedding_2nd_ls_k.device))

                # print(selected_embeds.shape)
                agg_embed = torch.cat(
                    [selected_embeds, selected_embeds_prime], dim=-1)
                agg_embed = torch.index_select(
                    agg_embed, -1, index=(torch.randperm(2*self.global_n_random)[:self.global_n_random]).to(agg_embed.device))

                global_random_embed[bi, pi] = agg_embed
        return global_random_embed

    def get_instance_embedding(self,
                               local_1, global_1,
                               local_2, global_2,
                               local_1_k, global_1_k,
                               local_2_k, global_2_k,
                               ):
        # local,globa: (nb, embed_dim, w,h,d)
        proj_instance_local_1 = self.instance_project_head_local(local_1)
        # pred_instacne_local_1 = self.instance_pred_head_local(proj_instance_local_1)
        pred_instance_local_1 = F.normalize(self.avgpool_local(
            proj_instance_local_1).view(proj_instance_local_1.size(0), -1), dim=1)

        proj_instance_local_2 = self.instance_project_head_local(local_2)
        # pred_instacne_local_2 = self.instance_pred_head_local(proj_instance_local_2)
        pred_instance_local_2 = F.normalize(self.avgpool_local(
            proj_instance_local_2).view(proj_instance_local_2.size(0), -1), dim=1)

        proj_instance_global_1 = self.instance_project_head_global(global_1)
        # proj_instance_global_1 = self.instance_pred_head_global(proj_instance_global_1)
        pred_instance_global_1 = F.normalize(self.avgpool_global(
            proj_instance_global_1).view(proj_instance_global_1.size(0), -1), dim=1)

        proj_instance_global_2 = self.instance_project_head_global(global_2)
        # proj_instance_global_2 = self.instance_pred_head_global(proj_instance_global_2)
        pred_instance_global_2 = F.normalize(self.avgpool_global(
            proj_instance_global_2).view(proj_instance_global_2.size(0), -1), dim=1)

        with torch.no_grad():
            proj_instance_local_1_k = self.instance_project_head_local_k(
                local_1_k)
            # pred_instacne_local_1 = self.instance_pred_head_local(proj_instance_local_1)
            pred_instance_local_1_k = F.normalize(self.avgpool_local(
                proj_instance_local_1_k).view(proj_instance_local_1_k.size(0), -1), dim=1)

            proj_instance_local_2_k = self.instance_project_head_local_k(
                local_2_k)
            # pred_instacne_local_2 = self.instance_pred_head_local(proj_instance_local_2)
            pred_instance_local_2_k = F.normalize(self.avgpool_local(
                proj_instance_local_2_k).view(proj_instance_local_2_k.size(0), -1), dim=1)

            proj_instance_global_1_k = self.instance_project_head_global_k(
                global_1_k)
            # proj_instance_global_1 = self.instance_pred_head_global(proj_instance_global_1)
            pred_instance_global_1_k = F.normalize(self.avgpool_global(
                proj_instance_global_1_k).view(proj_instance_global_1_k.size(0), -1), dim=1)

            proj_instance_global_2_k = self.instance_project_head_global_k(
                global_2_k)
            # proj_instance_global_2 = self.instance_pred_head_global(proj_instance_global_2)
            pred_instance_global_2_k = F.normalize(self.avgpool_global(
                proj_instance_global_2_k).view(proj_instance_global_2_k.size(0), -1), dim=1)

        return (pred_instance_local_1, pred_instance_global_1, pred_instance_local_2, pred_instance_global_2,
                pred_instance_local_1_k, pred_instance_global_1_k, pred_instance_local_2_k, pred_instance_global_2_k)

    def update_memory_bank(self, features_local, feature_global):
        batch_size = features_local.size(0)
        memory_size_local = self.memory_bank_local.size(0)

        ptr_local = int(self.memory_pointer_local.item())

        self.memory_bank_local = self.memory_bank_local.clone()
        self.memory_bank_local[ptr_local: ptr_local +
                               batch_size] = features_local
        self.memory_pointer_local = self.memory_pointer_local.clone()
        self.memory_pointer_local[0] = (
            ptr_local + batch_size) % memory_size_local

        memory_size_gloabl = self.memory_bank_global.size(0)
        ptr_global = int(self.memory_pointer_global.item())
        self.memory_bank_global = self.memory_bank_global.clone()
        self.memory_bank_global[ptr_global: ptr_global +
                                batch_size] = feature_global
        self.memory_pointer_global = self.memory_pointer_global.clone()
        self.memory_pointer_global[0] = (
            ptr_global + batch_size) % memory_size_gloabl

    def info_nce_instance(self, view_local_1, view_global_1, view_local_2, view_global_2,
                          temperature=0.5, global_loss_wt=0.5):
        # local,global : (nb, embed_dim)

        # nb, embed_dim) @nb, embed_dim) -> (nb, nb)
        pos_sim_local = torch.sum(
            view_local_1*view_local_2, dim=1).unsqueeze(1)/temperature
        pos_sim_global = torch.sum(
            view_global_1*view_global_2, dim=1).unsqueeze(1)/temperature

        negative_sim_local = torch.mm(
            view_local_1, self.memory_bank_local.T)/temperature
        negative_sim_global = torch.mm(
            view_global_1, self.memory_bank_global.T)/temperature

        logits_local = torch.cat([pos_sim_local, negative_sim_local], dim=1)
        logits_global = torch.cat([pos_sim_global, negative_sim_global], dim=1)

        labels = torch.arange(view_local_1.size(0)).to(view_local_1.device)

        loss_local = torch.nn.CrossEntropyLoss()(logits_local, labels)
        loss_global = torch.nn.CrossEntropyLoss()(logits_global, labels)

        loss = loss_local*(1-global_loss_wt) + loss_global*global_loss_wt
        return loss

    def training_step(self, batch, batch_idx):
        '''
        return ([local_embedding_1st_ls ,global_embedding_1st_ls],
            [local_embedding_2nd_ls ,global_embedding_2nd_ls],[local_loss,global_loss])
        pos should be in gpu_3
        '''

        [patch_ls, pos, masks] = batch
        mask1, mask2 = masks
        random_Crops, random_Crops_pair = patch_ls
        # torch.cuda.empty_cache()

        local_embedding_1st, global_embedding_1st = self.fpn(random_Crops)
        local_embedding_2nd, global_embedding_2nd = self.fpn(random_Crops_pair)

        if self.reconstruction_loss:
            recon1 = self.decoderlayer(local_embedding_1st)
            recon1 = self.transconv(recon1)
            recon1 = self.last_cnn(recon1)

            recon2 = self.decoderlayer(local_embedding_2nd)
            recon2 = self.transconv(recon2)
            recon2 = self.last_cnn(recon2)

        n_batch = random_Crops.shape[0]
        W, H, D = random_Crops.shape[-3:]

        # device = random_Crops.device

        local_embedding_1st_ls = torch.nn.functional.normalize(
            local_embedding_1st, p=2, dim=1)
        global_embedding_1st_ls = torch.nn.functional.normalize(
            global_embedding_1st, p=2, dim=1)

        local_embedding_2nd_ls = torch.nn.functional.normalize(
            local_embedding_2nd, p=2, dim=1)
        global_embedding_2nd_ls = torch.nn.functional.normalize(
            global_embedding_2nd, p=2, dim=1)

        # shape:  local_embedding_1st_ls,local_embedding_2nd_ls : shape (nb, embed_dim,patch_shape[0]//2,patch_shape[1]//2,patch_shape[2]//2)
        #        global_embedding_1st_ls, global_embedding_2nd_ls : shape (nb, embed_dim,patch_shape[0]//16,patch_shape[1]//16,patch_shape[2]//16)
        #                                                          or shape (nb, embed_dim,patch_shape[0]//16,patch_shape[1]//16,patch_shape[2]//4)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            local_embedding_1st_k, global_embedding_1st_k = self.fpn_k(
                random_Crops)
            local_embedding_2nd_k, global_embedding_2nd_k = self.fpn_k(
                random_Crops_pair)

            local_embedding_1st_ls_k = torch.nn.functional.normalize(
                local_embedding_1st_k, p=2, dim=1)
            global_embedding_1st_ls_k = torch.nn.functional.normalize(
                global_embedding_1st_k, p=2, dim=1)

            local_embedding_2nd_ls_k = torch.nn.functional.normalize(
                local_embedding_2nd_k, p=2, dim=1)
            global_embedding_2nd_ls_k = torch.nn.functional.normalize(
                global_embedding_2nd_k, p=2, dim=1)

        # make global embedding as same size of local embedding
        b, c, w, h, d = local_embedding_2nd_ls.shape  # x,y,z

        # make the coordinates of voxels of one anchor pair in the same axis.
        # shape (nb, n_pos_voxel,6)
        pos = pos.view(n_batch, self.n_pos_voxel, -1).to(dtype=torch.long)

        # get the anchor voxels embedding
        # anchor_global_1st_ls : (nb, n_pos_voxel, embedding_dim)
        # we do not need the momentum tensors to be here, because we extract the embeddings of anchor voxels
        (anchor_global_1st_ls, anchor_global_2nd_ls,
         anchor_local_1st_ls, anchor_local_2nd_ls), (pos_z) = self.get_local_global_anchors(local_embedding_1st_ls, local_embedding_2nd_ls,
                                                                                            global_embedding_1st_ls, global_embedding_2nd_ls, pos, n_batch)

        # get the similarity tensors of all voxels with anchor voxels
        # print(f"Similarity Matrix calculation started")

        # we need momentum encoder's tensor to compute similarity
        (sg, sg_prime, sl, sl_prime) = self.get_similarity_tensors(anchor_global_1st_ls,
                                                                   global_embedding_1st_ls_k, global_embedding_2nd_ls_k,
                                                                   anchor_local_1st_ls,
                                                                   local_embedding_1st_ls_k, local_embedding_2nd_ls_k
                                                                   )
        # print(f"Similarity Matrix calculation Ended")
        sl_x, sl_z, sl_y = sl.size()[-3:]
        sg_local = sg.clone()
        sg_prime_local = sg_prime.clone()
        # (nb, n_pos_voxel, patche_shape//2)
        sg_local = F.upsample(sg_local, size=(
            sl_x, sl_z, sl_y), mode='trilinear')

        sg_prime_local = F.upsample(sg_prime_local, size=(
            sl_x, sl_z, sl_y), mode='trilinear')  # (nb, n_pos_voxel, patche_shape//2)
        # get the negative global voxels embedding
        # neg_glbal_f_embed (n_batch,self.n_pos_voxel,self.embedding_dim,self.global_n_neg)
        # print(f"get_global_negative_voxels started")

        # we we need momentum encoder's tensor to extract negative voxels
        neg_glbal_f_embed = self.get_global_negative_voxels(sg, sg_prime, pos_z,
                                                            global_embedding_1st_ls_k, global_embedding_2nd_ls_k,
                                                            n_batch, b)
        # print(f"get_global_negative_voxels Ended")
        # get local negative voxel embedding
        # neg_local_f_embed :(n_batch,self.n_pos_voxel,self.embedding_dim,self.local_cand_neg)
        # print(f"get_hard_local_negative_voxels started")
        # we we need momentum encoder's tensor to extract negative voxels
        neg_local_f_embed = self.get_hard_local_negative_voxels(sl, sl_prime,
                                                                sg_local, sg_prime_local,
                                                                local_embedding_1st_ls_k, local_embedding_2nd_ls_k,
                                                                n_batch, pos, W, H, D, b)
        # print(f"get_hard_local_negative_voxels Ended")
        # get across-batch global negative voxel embeddings
        dtype_ = neg_local_f_embed.dtype
        # global_random_embed : (n_batch,self.n_pos_voxel,self.embedding_dim,self.global_n_random)
        # print(f"get_across_global_negative_voxels Started")
        # we we need momentum encoder's tensor to extract negative voxels
        global_random_embed = self.get_across_global_negative_voxels(global_embedding_1st_ls_k,
                                                                     global_embedding_2nd_ls_k,
                                                                     n_batch, dtype_)
        # print(f"get_across_global_negative_voxels Ened")
        # --- Get Info-NCE Loss---
        # get global info_nce_loss
        # print(f"get_global_info_nce_loss Started")
        # do not need momentum encoder tensors, as we have already compute negative samples from momentum encoder
        denom_global_loss, denom_global_loss_fi, label_global = self. get_global_info_nce_loss(global_random_embed, neg_glbal_f_embed,
                                                                                               anchor_global_1st_ls, anchor_global_2nd_ls,
                                                                                               n_batch)
        # print(f"get_global_info_nce_loss Ended")
        # get local info_nce_loss
        # previously it was local_loss_tot,loss_label
        # print(f"get_local_info_nce_loss Started")
        # do not need momentum encoder tensors, as we have already compute negative samples from momentum encoder
        denom_local_loss, denom_local_loss_fi, loss_label = self.get_local_info_nce_loss(anchor_local_1st_ls,
                                                                                         anchor_local_2nd_ls,
                                                                                         neg_local_f_embed, n_batch)
        # print(f"get_local_info_nce_loss Ended")

        # del neg_local_f_embed
        # del global_random_embed
        # del neg_glbal_f_embed
        # del grnd_global_neg_samples
        # need momentum encoder tensors, to compute momentum instance tensors
        local_1, global_1, local_2, global_2, local_1_k, global_1_k, local_2_k, global_2_k = self.get_instance_embedding(local_embedding_1st, global_embedding_1st,
                                                                                                                         local_embedding_2nd, global_embedding_2nd,
                                                                                                                         local_embedding_1st_k, global_embedding_1st_k,
                                                                                                                         local_embedding_2nd_k, global_embedding_2nd_k,
                                                                                                                         )

        instance_loss_1 = self.info_nce_instance(local_1, global_1, local_2_k, global_2_k,
                                                 temperature=self.model_kwargs['los_temp_instance'], global_loss_wt=0.5)
        instance_loss_2 = self.info_nce_instance(local_2, global_2, local_1_k, global_1_k,
                                                 temperature=self.model_kwargs['los_temp_instance'], global_loss_wt=0.5)
        instance_loss = (instance_loss_1 + instance_loss_2)/2

        self.update_memory_bank(local_1_k, global_1_k)

        local_loss_1 = torch.nn.CrossEntropyLoss()(denom_local_loss, loss_label)
        local_loss_2 = torch.nn.CrossEntropyLoss()(denom_local_loss_fi, loss_label)
        local_loss_tot = (local_loss_1 + local_loss_2)/2

        global_loss_1 = torch.nn.CrossEntropyLoss()(denom_global_loss, label_global)
        global_loss_2 = torch.nn.CrossEntropyLoss()(denom_global_loss_fi, label_global)
        global_loss_tot = (global_loss_1 + global_loss_2)/2

        loss = instance_loss + global_loss_tot + local_loss_tot
        if self.reconstruction_loss:

            reconst_loss = (self.compute_mse(random_Crops, recon1, mask1) +
                            self.compute_mse(random_Crops_pair, recon2, mask2))/2

            self.log('train_recons_loss', reconst_loss, on_step=True,
                     prog_bar=True, sync_dist=True, logger=True)

        self.log('train_local_loss', local_loss_tot, on_step=True,
                 prog_bar=True, sync_dist=True, logger=True)
        self.log('train_global_loss', global_loss_tot, on_step=True,
                 prog_bar=True, sync_dist=True, logger=True)
        self.log('train_instance_loss', instance_loss, on_step=True,
                 prog_bar=True, sync_dist=True, logger=True)
        if self.reconstruction_loss:
            return loss + reconst_loss
        return loss

    def compute_mse(self, ground, computed, mask):
        mse = torch.nn.MSELoss(
            size_average=None, reduce=None, reduction='none')(ground, computed)
        mask = mask.squeeze(1)
        mse = mse.squeeze(1)*mask
        mse = mse.sum(dim = (1,2,3))/ (mask.sum(dim = (1,2,3)) + 1e-8)
        mse = mse.mean()
        return mse

    def validation_step(self, batch, batch_idx):
        '''
        return ([local_embedding_1st_ls ,global_embedding_1st_ls],
            [local_embedding_2nd_ls ,global_embedding_2nd_ls],[local_loss,global_loss])
        pos should be in gpu_3
        '''

        [patch_ls, pos, masks] = batch
        mask1, mask2 = masks
        random_Crops, random_Crops_pair = patch_ls
        # torch.cuda.empty_cache()

        local_embedding_1st, global_embedding_1st = self.fpn(random_Crops)
        local_embedding_2nd, global_embedding_2nd = self.fpn(random_Crops_pair)

        if self.reconstruction_loss:
            recon1 = self.decoderlayer(local_embedding_1st)
            recon1 = self.transconv(recon1)
            recon1 = self.last_cnn(recon1)

            recon2 = self.decoderlayer(local_embedding_2nd)
            recon2 = self.transconv(recon2)
            recon2 = self.last_cnn(recon2)

        n_batch = random_Crops.shape[0]
        W, H, D = random_Crops.shape[-3:]

        # device = random_Crops.device

        local_embedding_1st_ls = torch.nn.functional.normalize(
            local_embedding_1st, p=2, dim=1)
        global_embedding_1st_ls = torch.nn.functional.normalize(
            global_embedding_1st, p=2, dim=1)

        local_embedding_2nd_ls = torch.nn.functional.normalize(
            local_embedding_2nd, p=2, dim=1)
        global_embedding_2nd_ls = torch.nn.functional.normalize(
            global_embedding_2nd, p=2, dim=1)

        # shape:  local_embedding_1st_ls,local_embedding_2nd_ls : shape (nb, embed_dim,patch_shape[0]//2,patch_shape[1]//2,patch_shape[2]//2)
        #        global_embedding_1st_ls, global_embedding_2nd_ls : shape (nb, embed_dim,patch_shape[0]//16,patch_shape[1]//16,patch_shape[2]//16)
        #                                                          or shape (nb, embed_dim,patch_shape[0]//16,patch_shape[1]//16,patch_shape[2]//4)
        with torch.no_grad():
            self._momentum_update_key_encoder()
            local_embedding_1st_k, global_embedding_1st_k = self.fpn_k(
                random_Crops)
            local_embedding_2nd_k, global_embedding_2nd_k = self.fpn_k(
                random_Crops_pair)

            local_embedding_1st_ls_k = torch.nn.functional.normalize(
                local_embedding_1st_k, p=2, dim=1)
            global_embedding_1st_ls_k = torch.nn.functional.normalize(
                global_embedding_1st_k, p=2, dim=1)

            local_embedding_2nd_ls_k = torch.nn.functional.normalize(
                local_embedding_2nd_k, p=2, dim=1)
            global_embedding_2nd_ls_k = torch.nn.functional.normalize(
                global_embedding_2nd_k, p=2, dim=1)

        # make global embedding as same size of local embedding
        b, c, w, h, d = local_embedding_2nd_ls.shape  # x,y,z

        # make the coordinates of voxels of one anchor pair in the same axis.
        # shape (nb, n_pos_voxel,6)
        pos = pos.view(n_batch, self.n_pos_voxel, -1).to(dtype=torch.long)

        # get the anchor voxels embedding
        # anchor_global_1st_ls : (nb, n_pos_voxel, embedding_dim)
        # we do not need the momentum tensors to be here, because we extract the embeddings of anchor voxels
        (anchor_global_1st_ls, anchor_global_2nd_ls,
         anchor_local_1st_ls, anchor_local_2nd_ls), (pos_z) = self.get_local_global_anchors(local_embedding_1st_ls, local_embedding_2nd_ls,
                                                                                            global_embedding_1st_ls, global_embedding_2nd_ls, pos, n_batch)

        # get the similarity tensors of all voxels with anchor voxels
        # print(f"Similarity Matrix calculation started")

        # we need momentum encoder's tensor to compute similarity
        (sg, sg_prime, sl, sl_prime) = self.get_similarity_tensors(anchor_global_1st_ls,
                                                                   global_embedding_1st_ls_k, global_embedding_2nd_ls_k,
                                                                   anchor_local_1st_ls,
                                                                   local_embedding_1st_ls_k, local_embedding_2nd_ls_k
                                                                   )
        # print(f"Similarity Matrix calculation Ended")
        sl_x, sl_z, sl_y = sl.size()[-3:]
        sg_local = sg.clone()
        sg_prime_local = sg_prime.clone()
        # (nb, n_pos_voxel, patche_shape//2)
        sg_local = F.upsample(sg_local, size=(
            sl_x, sl_z, sl_y), mode='trilinear')

        sg_prime_local = F.upsample(sg_prime_local, size=(
            sl_x, sl_z, sl_y), mode='trilinear')  # (nb, n_pos_voxel, patche_shape//2)
        # get the negative global voxels embedding
        # neg_glbal_f_embed (n_batch,self.n_pos_voxel,self.embedding_dim,self.global_n_neg)
        # print(f"get_global_negative_voxels started")

        # we we need momentum encoder's tensor to extract negative voxels
        neg_glbal_f_embed = self.get_global_negative_voxels(sg, sg_prime, pos_z,
                                                            global_embedding_1st_ls_k, global_embedding_2nd_ls_k,
                                                            n_batch, b)
        # print(f"get_global_negative_voxels Ended")
        # get local negative voxel embedding
        # neg_local_f_embed :(n_batch,self.n_pos_voxel,self.embedding_dim,self.local_cand_neg)
        # print(f"get_hard_local_negative_voxels started")
        # we we need momentum encoder's tensor to extract negative voxels
        neg_local_f_embed = self.get_hard_local_negative_voxels(sl, sl_prime,
                                                                sg_local, sg_prime_local,
                                                                local_embedding_1st_ls_k, local_embedding_2nd_ls_k,
                                                                n_batch, pos, W, H, D, b)
        # print(f"get_hard_local_negative_voxels Ended")
        # get across-batch global negative voxel embeddings
        dtype_ = neg_local_f_embed.dtype
        # global_random_embed : (n_batch,self.n_pos_voxel,self.embedding_dim,self.global_n_random)
        # print(f"get_across_global_negative_voxels Started")
        # we we need momentum encoder's tensor to extract negative voxels
        global_random_embed = self.get_across_global_negative_voxels(global_embedding_1st_ls_k,
                                                                     global_embedding_2nd_ls_k,
                                                                     n_batch, dtype_)
        # print(f"get_across_global_negative_voxels Ened")
        # --- Get Info-NCE Loss---
        # get global info_nce_loss
        # print(f"get_global_info_nce_loss Started")
        # do not need momentum encoder tensors, as we have already compute negative samples from momentum encoder
        denom_global_loss, denom_global_loss_fi, label_global = self. get_global_info_nce_loss(global_random_embed, neg_glbal_f_embed,
                                                                                               anchor_global_1st_ls, anchor_global_2nd_ls,
                                                                                               n_batch)
        # print(f"get_global_info_nce_loss Ended")
        # get local info_nce_loss
        # previously it was local_loss_tot,loss_label
        # print(f"get_local_info_nce_loss Started")
        # do not need momentum encoder tensors, as we have already compute negative samples from momentum encoder
        denom_local_loss, denom_local_loss_fi, loss_label = self.get_local_info_nce_loss(anchor_local_1st_ls,
                                                                                         anchor_local_2nd_ls,
                                                                                         neg_local_f_embed, n_batch)
        # print(f"get_local_info_nce_loss Ended")

        # del neg_local_f_embed
        # del global_random_embed
        # del neg_glbal_f_embed
        # del grnd_global_neg_samples
        # need momentum encoder tensors, to compute momentum instance tensors
        local_1, global_1, local_2, global_2, local_1_k, global_1_k, local_2_k, global_2_k = self.get_instance_embedding(local_embedding_1st, global_embedding_1st,
                                                                                                                         local_embedding_2nd, global_embedding_2nd,
                                                                                                                         local_embedding_1st_k, global_embedding_1st_k,
                                                                                                                         local_embedding_2nd_k, global_embedding_2nd_k,
                                                                                                                         )

        instance_loss_1 = self.info_nce_instance(local_1, global_1, local_2_k, global_2_k,
                                                 temperature=self.model_kwargs['los_temp_instance'], global_loss_wt=0.5)
        instance_loss_2 = self.info_nce_instance(local_2, global_2, local_1_k, global_1_k,
                                                 temperature=self.model_kwargs['los_temp_instance'], global_loss_wt=0.5)
        instance_loss = (instance_loss_1 + instance_loss_2)/2

        self.update_memory_bank(local_1_k, global_1_k)

        local_loss_1 = torch.nn.CrossEntropyLoss()(denom_local_loss, loss_label)
        local_loss_2 = torch.nn.CrossEntropyLoss()(denom_local_loss_fi, loss_label)
        local_loss_tot = (local_loss_1 + local_loss_2)/2

        global_loss_1 = torch.nn.CrossEntropyLoss()(denom_global_loss, label_global)
        global_loss_2 = torch.nn.CrossEntropyLoss()(denom_global_loss_fi, label_global)
        global_loss_tot = (global_loss_1 + global_loss_2)/2

        loss = instance_loss + global_loss_tot + local_loss_tot
        if self.reconstruction_loss:

            reconst_loss = (self.compute_mse(random_Crops, recon1, mask1) +
                            self.compute_mse(random_Crops_pair, recon2, mask2))/2

            self.log('val_recons_loss', reconst_loss, on_step=True,
                     prog_bar=True, sync_dist=True, logger=True)

        self.log('val_local_loss', local_loss_tot, on_step=True,
                 prog_bar=True, sync_dist=True, logger=True)
        self.log('val_global_loss', global_loss_tot, on_step=True,
                 prog_bar=True, sync_dist=True, logger=True)
        self.log('val_instance_loss', instance_loss, on_step=True,
                 prog_bar=True, sync_dist=True, logger=True)
        if self.reconstruction_loss:
            return loss + reconst_loss
        return loss


def init_params(m):

    if type(m) == nn.Linear or type(m) == nn.Conv3d:
        # Random weight initialisation
        m.weight.data = torch.randn(m.weight.size())
        # m.bias.data=torch.zeros(m.bias.size())


# def get_model_for_inference(weight_path, load_pretrained_wt=True, fully_random_weight=False,
#                             apply_z_stride=[2, 1, 1, 2]):

#     n_batch = 5
#     n_pos_voxel = 50
#     global_n_neg = 2
#     local_cand_neg = 2
#     global_n_random = 2
#     embedding_dim = 128
#     los_temp = 0.5

#     radius = 1

#     device = torch.device('cpu')

#     model_kwargs = {'n_batch': n_batch, 'n_pos_voxel': n_pos_voxel, 'global_n_neg': global_n_neg,
#                     'local_cand_neg': local_cand_neg, 'embedding_dim': embedding_dim, 'global_n_random': global_n_random,
#                     'los_temp': los_temp, 'device': device, 'radius': radius, 'apply_z_stride': apply_z_stride,
#                     }
#     model = Model(model_kwargs)

#     # wt_path = "<path/to>/encoder_weights/checkpoint.pt"
#     if load_pretrained_wt:
#         print("Loading pretrained Model weight")

#         checkpoint = torch.load(weight_path, map_location=device)
#         if 'epoch' not in checkpoint and 'steps' not in checkpoint:  # for older version of saved models
#             print(f"Loading from Old versions")
#             model.load_state_dict(checkpoint)
#             # sanity check
#             for name, param in model.named_parameters():
#                 print(name, param.isnan().any(), torch.eq(
#                     param, (checkpoint[name].to(model.device))).all())
#         else:
#             # check_point_path = os.path.join(check_dir,check_point_name)
#             print(f"Loading from new versions")
#             model.load_state_dict(checkpoint['model_state_dict'])
#             for name, param in model.named_parameters():
#                 print(name, param.isnan().any(), torch.eq(
#                     param, (checkpoint['model_state_dict'][name].to(model.device))).all())
#         # check the
#     elif fully_random_weight:
#         print("No weight is loaded, randomly setting weights")
#         model.apply(init_params)

#     else:
#         print("No weight is loaded, weights are set to default values")

#     device = torch.device("cpu")
#     model.to(device)
#     model.eval()

#     return model


def get_model_for_inference(weight_path, load_pretrained_wt=True, fully_random_weight=False,
                            apply_z_stride=[2, 1, 1, 2], model_type='vanila'):

    model = models.get_model(args, model_type=model_type)

    n_batch = 5
    n_pos_voxel = 50
    global_n_neg = 2
    local_cand_neg = 2
    global_n_random = 2
    embedding_dim = 128
    los_temp = 0.5

    radius = 1

    device = torch.device('cpu')

    model_kwargs = {'n_batch': n_batch, 'n_pos_voxel': n_pos_voxel, 'global_n_neg': global_n_neg,
                    'local_cand_neg': local_cand_neg, 'embedding_dim': embedding_dim, 'global_n_random': global_n_random,
                    'los_temp': los_temp, 'device': device, 'radius': radius, 'apply_z_stride': apply_z_stride,
                    }
    model = Model(model_kwargs)

    args = main.get_args(config_file_lambda="experiment_model_3_new_1",
                         device=torch.device('cpu'),
                         use_config_obj=True,
                         config_obj_name=experiment_configurations.conf_model_4_exp_3())
    train_experiment(args, model_type='moA')

    # wt_path = "<path/to>/encoder_weights/checkpoint.pt"
    if load_pretrained_wt:
        print("Loading pretrained Model weight")

        checkpoint = torch.load(weight_path, map_location=device)
        if 'epoch' not in checkpoint and 'steps' not in checkpoint:  # for older version of saved models
            print(f"Loading from Old versions")
            model.load_state_dict(checkpoint)
            # sanity check
            for name, param in model.named_parameters():
                print(name, param.isnan().any(), torch.eq(
                    param, (checkpoint[name].to(model.device))).all())
        else:
            # check_point_path = os.path.join(check_dir,check_point_name)
            print(f"Loading from new versions")
            model.load_state_dict(checkpoint['model_state_dict'])
            for name, param in model.named_parameters():
                print(name, param.isnan().any(), torch.eq(
                    param, (checkpoint['model_state_dict'][name].to(model.device))).all())
        # check the
    elif fully_random_weight:
        print("No weight is loaded, randomly setting weights")
        model.apply(init_params)

    else:
        print("No weight is loaded, weights are set to default values")

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    return model


class voxel_classifier(pl.LightningModule):
    def __init__(self, pretrained_model, input_dim=128, out_dim=16):
        '''
        pretrained_model is fully loaded with trained model.

        '''
        super().__init__()

        self.base_model = pretrained_model
        self.out_dim = out_dim

        # make the base_model untrainable
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.lin = torch.nn.Linear(input_dim, out_dim)

    def forward(self, x, original_shape):
        D, W, H = tuple(x.shape[-3:])

        local_x, global_x = self.base_model(x)
        # print(f"local_x : {local_x.shape}")
        # print(f"global_x : {global_x.shape}")
        local_x = torch.nn.functional.upsample(
            local_x, size=(D, W, H), mode='trilinear')
        global_x = torch.nn.functional.upsample(
            global_x, size=(D, W, H), mode='trilinear')

        # print(f"uplocal_x : {local_x.shape}")
        # print(f"upglobal_x : {global_x.shape}")

        x = local_x + global_x
        x = F.normalize(x, p=2.0, dim=1)
        # print(f"combined : {x.shape}")

        # discard the padding added for convolution
        x_pad, y_pad, z_pad = x.shape[2] - original_shape[0],  x.shape[3] - \
            original_shape[1], x.shape[4] - original_shape[2]
        # print(f"original shape: {original_shape}")
        # print(f"x_pad,y_pad,z_pad: {x_pad,y_pad,z_pad}")
        x_pre_post, y_pre_post, z_pre_post = [(int(p/2), int(p/2)) if p % 2 == 0 else (
            int((p+1)/2), int(p - (p+1)/2)) for p in [x_pad, y_pad, z_pad]]
        x_len, y_len, z_len = tuple(x.shape[-3:])
        x = x[:, :, x_pre_post[0]:x_len-x_pre_post[1], y_pre_post[0]
            :y_len-y_pre_post[1], z_pre_post[0]:z_len-z_pre_post[1]]

        # print(f"after depadding, x.shape : {x.shape}")
        x = torch.permute(x, (0, 2, 3, 4, 1))
        out = self.lin(x)

        return out


# def get_pretrained_model_voxel_classification_model(pretrained_weight_name, apply_z_stride=[2, 2, 2, 2],
#                                                     load_pretrained_wt=True, fully_random_weight=False, full_wt_path_given=False):

#     if not full_wt_path_given:
#         print(f"Full weight path is not given")
#         pretrained_model_weight_path = os.path.join(
#             config.model_weight_path(), pretrained_weight_name)
#     else:
#         pretrained_model_weight_path = pretrained_weight_name
#     print(f"weight path: {pretrained_model_weight_path}")
#     # load_pretrained_wt = True
#     # fully_random_weight= False
#     apply_z_stride = apply_z_stride
#     pretrained_model = get_model_for_inference(weight_path=pretrained_model_weight_path,
#                                                load_pretrained_wt=load_pretrained_wt,
#                                                fully_random_weight=fully_random_weight,
#                                                apply_z_stride=apply_z_stride)

#     return pretrained_model.fpn


def reinitialize_weight(origianl_model):
    model = copy.deepcopy(origianl_model)
    for layer in model.modules():
        if isinstance(layer, (nn.Linear, nn.Conv3d)):
            init.kaiming_uniform_(layer.weight, nonlinearity='relu')
        elif isinstance(layer, nn.BatchNorm3d):
            init.constant_(layer.weight, 1)
            init.constant_(layer.bias, 0)
    return model


def get_pretrained_model_voxel_classification_model(pretrained_weight_name,
                                                    apply_z_stride=[
                                                        2, 2, 2, 2],
                                                    load_pretrained_wt=True,
                                                    fully_random_weight=False,
                                                    full_wt_path_given=False, **kwargs):

    if not full_wt_path_given:
        print(f"Full weight path is not given")
        pretrained_model_weight_path = os.path.join(
            config.model_weight_path(), pretrained_weight_name)
    else:
        pretrained_model_weight_path = pretrained_weight_name
    print(f"weight path: {pretrained_model_weight_path}")

    if 'model_type' in kwargs and 'model_kwargs' in kwargs:
        model = get_model(kwargs['model_kwargs'],
                          model_type=kwargs['model_type'])
        if fully_random_weight:
            print('Randomly initialized weights are being loaded')
            model = reinitialize_weight(model.fpn)
            return model
        checkpoint = torch.load(
            pretrained_model_weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'])

        return model.fpn

    else:
        # load_pretrained_wt = True
        # fully_random_weight= False
        apply_z_stride = apply_z_stride
        pretrained_model = get_model_for_inference(weight_path=pretrained_model_weight_path,
                                                   load_pretrained_wt=load_pretrained_wt,
                                                   fully_random_weight=fully_random_weight,
                                                   apply_z_stride=apply_z_stride)

        return pretrained_model.fpn


"""
pretrained_weight_name = 'backup_model_3_stride_96_96_96_s_-1_p_50_llf0_loss_7.428323824853356.pt'
pretrained_model = get_pretrained_model_voxel_classification_model(pretrained_weight_name = pretrained_weight_name,apply_z_stride = [2,2,2,2])
"""


class voxel_age_prediction(pl.LightningModule):
    def __init__(self, pretrained_model, input_dim=128, out_dim=1, bias=True):
        '''
        pretrained_model is fully loaded with trained model.

        '''
        super().__init__()

        self.base_model = pretrained_model
        self.out_dim = out_dim

        # make the base_model untrainable
        for param in self.base_model.parameters():
            param.requires_grad = False

        self.lin = torch.nn.Linear(input_dim, out_dim, bias=bias)

    def forward(self, x, original_shape, apply_reg=True):
        D, W, H = tuple(x[0].shape[-3:]) if len(x > 0) else tuple(x.shape[-3:])

        if isinstance(original_shape, list):
            original_shape = original_shape[0]

        local_x, global_x = self.base_model(x)
        # print(f"local_x : {local_x.shape}")
        # print(f"global_x : {global_x.shape}")
        local_x = torch.nn.functional.upsample(
            local_x, size=(D, W, H), mode='trilinear')
        global_x = torch.nn.functional.upsample(
            global_x, size=(D, W, H), mode='trilinear')
        # print(f"uplocal_x : {local_x.shape}")
        # print(f"upglobal_x : {global_x.shape}")

        x = local_x + global_x
        x = F.normalize(x, p=2.0, dim=1)
        # print(f"combined : {x.shape}")

        # discard the padding added for convolution
        x_pad, y_pad, z_pad = x.shape[2] - original_shape[0],  x.shape[3] - \
            original_shape[1], x.shape[4] - original_shape[2]
        # print(f"original shape: {original_shape}")
        # print(f"x_pad,y_pad,z_pad: {x_pad,y_pad,z_pad}")
        x_pre_post, y_pre_post, z_pre_post = [(int(p/2), int(p/2)) if p % 2 == 0 else (
            int((p+1)/2), int(p - (p+1)/2)) for p in [x_pad, y_pad, z_pad]]
        x_len, y_len, z_len = tuple(x.shape[-3:])
        x = x[:, :, x_pre_post[0]:x_len-x_pre_post[1], y_pre_post[0]
            :y_len-y_pre_post[1], z_pre_post[0]:z_len-z_pre_post[1]]

        # print(f"after depadding, x.shape : {x.shape}")
        x = torch.permute(x, (0, 2, 3, 4, 1))
        if apply_reg:
            out = self.lin(x)
            return out
        return x


class age_prediction_via_attention(pl.LightningModule):
    def __init__(self, pretrained_model, input_dim=128, out_dim=1, bias=True):
        '''
        pretrained_model is fully loaded with trained model.

        '''
        super().__init__()

        self.base_model = pretrained_model
        self.out_dim = out_dim

        # make the base_model untrainable
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.projection = torch.nn.Linear(input_dim, input_dim)
        self.proj_act = torch.nn.ReLU()
        self.attention = torch.nn.Linear(input_dim, 1, bias=bias)
        self.attn_act = torch.nn.ReLU()
        # self.atten_soft = torch.nn.Softmax(dim=-1)
        self.lin = torch.nn.Linear(input_dim, out_dim, bias=bias)

    def forward(self, x, original_shape, apply_reg=True):
        D, W, H = tuple(x[0].shape[-3:]) if len(x > 0) else tuple(x.shape[-3:])

        if isinstance(original_shape, list):
            original_shape = original_shape[0]

        local_x, global_x = self.base_model(x)
        # print(f"local_x : {local_x.shape}")
        # print(f"global_x : {global_x.shape}")
        local_x = torch.nn.functional.upsample(
            local_x, size=(D, W, H), mode='trilinear')
        global_x = torch.nn.functional.upsample(
            global_x, size=(D, W, H), mode='trilinear')
        # print(f"uplocal_x : {local_x.shape}")
        # print(f"upglobal_x : {global_x.shape}")

        x = local_x + global_x
        x = F.normalize(x, p=2.0, dim=1)
        # print(f"combined : {x.shape}")

        # discard the padding added for convolution
        x_pad, y_pad, z_pad = x.shape[2] - original_shape[0],  x.shape[3] - \
            original_shape[1], x.shape[4] - original_shape[2]
        # print(f"original shape: {original_shape}")
        # print(f"x_pad,y_pad,z_pad: {x_pad,y_pad,z_pad}")
        x_pre_post, y_pre_post, z_pre_post = [(int(p/2), int(p/2)) if p % 2 == 0 else (
            int((p+1)/2), int(p - (p+1)/2)) for p in [x_pad, y_pad, z_pad]]
        x_len, y_len, z_len = tuple(x.shape[-3:])
        x = x[:, :, x_pre_post[0]:x_len-x_pre_post[1], y_pre_post[0]
            :y_len-y_pre_post[1], z_pre_post[0]:z_len-z_pre_post[1]]

        # print(f"after depadding, x.shape : {x.shape}")
        x = torch.permute(x, (0, 2, 3, 4, 1))
        b, h, w, l, c = x.shape
        x = x.reshape(b, -1, c)

        x_atten = self.proj_act(self.projection(x))
        # print(f"after projection: {x_atten.shape}")

        x_atten = self.attn_act(self.attention(x_atten))
        # print(f"after x_atten: {x_atten.shape}")

        x = (x*x_atten).sum(dim=1)
        # print(f"after x*x_atten: {x.shape}")

        if apply_reg:
            out = self.lin(x)
            return out, x_atten
        return x, x_atten


class Voxel_Age_Regressor(pl.LightningModule):
    def __init__(self, input_dim=128, out_dim=1, scaling=10, bias=True):
        '''
        Implemented a non linear classifier

        '''
        super().__init__()

        self.projection1 = torch.nn.Linear(input_dim, input_dim*scaling)
        self.bn1 = torch.nn.BatchNorm1d(input_dim*scaling)
        self.proj_act1 = torch.nn.ReLU()

        self.projection2 = torch.nn.Linear(
            input_dim*scaling, input_dim*scaling)
        self.bn2 = torch.nn.BatchNorm1d(input_dim*scaling)
        self.proj_act2 = torch.nn.ReLU()

        self.projection3 = torch.nn.Linear(
            input_dim*scaling, input_dim*scaling)
        self.bn3 = torch.nn.BatchNorm1d(input_dim*scaling)
        self.proj_act3 = torch.nn.ReLU()

        self.lin = torch.nn.Linear(input_dim*scaling, out_dim)

    def forward(self, x):
        ''' x: 2D vector. (batch, input_dim)
        '''
        x = self.projection1(x)
        x = self.bn1(x)
        x = self.proj_act1(x)

        x = self.projection2(x)
        x = self.bn2(x)
        x = self.proj_act2(x)

        x = self.projection3(x)
        x = self.bn3(x)
        x = self.proj_act3(x)

        x = self.lin(x)
        return x


def get_pretrained_model_age_prediction_model(pretrained_weight_name, apply_z_stride=[2, 2, 2, 2]):

    pretrained_model_weight_path = os.path.join(
        config.model_weight_path(), pretrained_weight_name)

    load_pretrained_wt = True
    fully_random_weight = False
    apply_z_stride = apply_z_stride
    pretrained_model = get_model_for_inference(weight_path=pretrained_model_weight_path,
                                               load_pretrained_wt=load_pretrained_wt,
                                               fully_random_weight=fully_random_weight,
                                               apply_z_stride=apply_z_stride)
    return pretrained_model.fpn


def get_trained_model_with_ckpoint_loaded(model_name_string, chkpoint_path=None,
                                          pretrained_weight_name=None, device=torch.device('cpu')):
    '''
    Load a model for specific downstream task with checkpoint
    Arguments:

        model_name_string: sting. code word for the downstream task.
        chkpoint_path: full path of the checkpoint
        pretrained_weight_name: the checkpoint for main voxel embedding network.

    Return:
        the checkpont loaded model


    '''
    if model_name_string == 'voxel_age_prediction':
        # pretrained_weight_name = 'test_experiment_model_3_new_1_model_3_96_96_96_s_-1_p_50_llf0_loss_7.455170304167504.pt'
        pretrained_model = get_pretrained_model_voxel_classification_model(pretrained_weight_name, apply_z_stride=[2, 2, 2, 2],
                                                                           load_pretrained_wt=True, fully_random_weight=False)

        # load full age prediction model
        input_dim = 128
        out_dim = 1
        model = voxel_age_prediction(
            pretrained_model, input_dim=input_dim, out_dim=out_dim)
        if chkpoint_path is not None:
            checkpoint = torch.load(chkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])

        return model


"""

model_name_string = 'voxel_age_prediction'
checkpoint_name = None
chkpoint_path = None
pretrained_weight_name = 'test_experiment_model_3_new_1_model_3_96_96_96_s_-1_p_50_llf0_loss_7.455170304167504.pt'
model = get_trained_model_with_ckpoint_loaded(model_name_string, chkpoint_path, pretrained_weight_name = None,device = torch.device('cpu'))

"""

model_dict = {'default': Model,
              'voxel_infonce_encoder': Model,
              }
