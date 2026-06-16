import torch

import lightning.pytorch as pl

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import preprocessing
import nibabel as nib
import random
import config_pixpro


def get_train_dataloader(type_dataloader='train',
                         shape=(96, 96, 96),
                         min_overlap_fraction=0.2, max_overlap_fraction=0.8):
    '''Convenient method to get the train dataset'''

    if type_dataloader == 'train':

        dataset = MRIDataset(csv_path=config_pixpro.train_img_paths, shape=shape,
                             min_overlap_fraction=min_overlap_fraction,
                             max_overlap_fraction=max_overlap_fraction)
        return dataset
    else:

        dataset = MRIDataset(csv_path=config_pixpro.val_img_paths, shape=shape,
                             min_overlap_fraction=min_overlap_fraction,
                             max_overlap_fraction=max_overlap_fraction)
        return dataset

class MRIDataset(Dataset):
    def __init__(self, csv_path, shape=(96, 96, 96), file_type="T1_brain.nii.gz",
                 min_overlap_fraction=0.2, max_overlap_fraction=0.8):
        """
        Args:
            csv_path (str): Path to the CSV file containing MRI paths.
            shape (tuple): Shape of the cube to sample (default: (96, 96, 96)).
            min_overlap_fraction (float): Minimum overlap fraction between cubes.
            max_overlap_fraction (float): Maximum overlap fraction between cubes.
        """

        self.data = csv_path
        self.shape = shape
        self.file_type = file_type
        self.min_overlap_fraction = min_overlap_fraction
        self.max_overlap_fraction = max_overlap_fraction

    def __len__(self):
        return len(self.data)

    def __sample_cube(self, img, mask):
        """
        Samples a cube from the given image and ensures it satisfies the foreground condition.
        Args:
            img (ndarray): 3D MRI image.
            mask (ndarray): Foreground mask.

        Returns:
            cube, coordinates, foreground mask
        """

        D, H, W = img.shape
        d, h, w = self.shape

        while True:
            z_start = random.randint(0, D-d)
            y_start = random.randint(0, H-h)
            x_start = random.randint(0, W-w)

            z_end, y_end, x_end = z_start + d, y_start+h, x_start + w

            cube = img[z_start:z_end, y_start:y_end, x_start:x_end]
            cube_mask = mask[z_start:z_end, y_start:y_end, x_start:x_end]

            if np.sum(cube_mask) >= 0.5 * d * h * w:
                return cube, (z_start, y_start, x_start, z_end, y_end, x_end), cube_mask

    def __getitem__(self, idx):
        img_path = self.data[idx]

        img, img_mask = preprocessing.get_torch_from_single_nifti(nifti_dir=img_path, file_type=self.file_type,
                                                                  n_sample=1, save_numpy=False,
                                                                  save_numpy_dir=None)
        img = img.squeeze()
        img = img.numpy()
        mask = (img != 0).astype(np.float32)

        cube1, coord1, mask1 = self.__sample_cube(img, mask)
        cube2, coord2, mask2 = None, None, None

        while True:
            overlap_fraction = random.uniform(
                self.min_overlap_fraction, self.max_overlap_fraction)
            #print(f"overlap_fraction: {overlap_fraction}")
            z_overlap = int(overlap_fraction*self.shape[0])
            y_overlap = int(overlap_fraction*self.shape[1])
            x_overlap = int(overlap_fraction*self.shape[2])

            z_shift = random.randint(-z_overlap, z_overlap)
            y_shift = random.randint(-y_overlap, y_overlap)
            x_shift = random.randint(-x_overlap, x_overlap)

            z_start = max(coord1[0] + z_shift, 0)
            #print(f"z_start: {z_start}")
            y_start = max(coord1[1]+y_shift, 0)
            #print(f"y_start: {y_start}")
            x_start = max(coord1[2]+x_shift, 0)
            #print(f"x_start: {x_start}")
            z_end, y_end, x_end = z_start + \
                self.shape[0], y_start + self.shape[1], x_start + self.shape[2]

            if z_end > img.shape[0] or y_end > img.shape[1] or x_end > img.shape[2]:
                continue

            cube2 = img[z_start:z_end, y_start:y_end, x_start:x_end]
            mask2 = mask[z_start:z_end, y_start:y_end, x_start:x_end]

            mask1 = (cube1 != 0).astype(bool)
            mask2 = (cube2 != 0).astype(bool)
            overlap_mask = (mask1 & mask2)
            if np.sum(mask2) >= 0.5 * np.prod(self.shape) and np.sum(overlap_mask) >= 0.2 * np.prod(self.shape):
                coord2 = (z_start, y_start, x_start, z_end, y_end, x_end)
                break

        return (

            torch.tensor(cube1, dtype=torch.float32).unsqueeze(0),
            torch.tensor(cube2, dtype=torch.float32).unsqueeze(0),
            torch.tensor(coord1, dtype=torch.float32),
            torch.tensor(coord2, dtype=torch.float32),
            torch.tensor(mask1, dtype=torch.float32),
            torch.tensor(mask2, dtype=torch.float32)

        )


def get_dataloader(batch_size=3,
                   type_dataloader='train',
                   shape=(96, 96, 96),
                   min_overlap_fraction=0.2, max_overlap_fraction=0.8):

    dataset = get_train_dataloader(type_dataloader=type_dataloader,
                                                     shape=shape,
                                                     min_overlap_fraction=min_overlap_fraction, max_overlap_fraction=max_overlap_fraction)

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True if type_dataloader == 'train' else False,
                                             num_workers=5,
                                             )
    return dataloader


class PixPro3DDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=3,
                 patch_size=(96, 96, 96),
                 min_overlap_fraction=0.2, max_overlap_fraction=0.8):
        super().__init__()

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.min_overlap_fraction = min_overlap_fraction
        self.max_overlap_fraction = max_overlap_fraction

    def setup(self, stage =None):
        
        if stage == 'fit' or stage is None:
            self.train_dataset = get_train_dataloader(type_dataloader='train',
                         shape=self.patch_size,
                         min_overlap_fraction=self.min_overlap_fraction,
                         max_overlap_fraction= self.max_overlap_fraction)
            
            self.val_dataset = get_train_dataloader(type_dataloader='val',
                         shape=self.patch_size,
                         min_overlap_fraction=self.min_overlap_fraction,
                         max_overlap_fraction= self.max_overlap_fraction)            
        
        
        
        
        
        
    def train_dataloader(self):
                
        return DataLoader(self.train_dataset, 
                          batch_size = self.batch_size,
                          shuffle = True,
                          num_workers = 0)

    def val_dataloader(self):

        return DataLoader(self.val_dataset, 
                          batch_size = self.batch_size,
                          shuffle = False,
                          num_workers = 0)
