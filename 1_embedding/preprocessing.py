'''
Came from main_v2.
Problem in main_v2:
    (1) loss still does not merge
Changes in main_v3:
    (1) make the positive voxels in global similarity INF prior to upsample (done, but no improvement)
    (2) Possibly try to select random samples not from pixels that are closer than a certain distance.
    (3) adding symmetric loss (done,  but no improvement) (no improvement)
    (4) in nifti image format the last dimension is z dimension, we changed our network to accomodate this fact. (No improvement)
    (5) The crucial thing is the following:
            when patches does not overlap, we sample half of the positive voxel from first patch
            and the rest half from the second patch.
            THIS IS WRONG.
            by no means we can not tell the voxel in 1st patch would match the voxel in the
            second patch. we need to correct that.
        we will do that in the next iteration.

changes done in main_v4.py
    
    added the feaure normalization of embedding before calculating the similarity. And now local and global losses are decreasing
    But the model OOM error is shown after some epochs. May be saving the computational graph which should not be saved.

Changes done in main_v5.py

adding model parralism (but did not work yet.)


Changes done in main_v6.py
    adding new methods to sample from only the foreground (done)
    now need to change data preprocessing codes to accomodate change

'''


import zipfile
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
import shutil
from GPUtil import showUtilization as gpu_usage
import time
import numpy as np
import random
import itertools
import pandas as pd
# add few things


import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import nibabel as nib
from zipfile import ZipFile
import random
from operator import mul
from functools import reduce

import lightning.pytorch as pl

# ---------------------------------------------------------------------------
# REPRODUCIBILITY: MONAI's Randomizable base seeds `self.R` from OS entropy at
# construction, and `set_random_state` is never called in this stack. Seeding
# python/numpy/torch therefore does NOT reach the patch-centre sampler, which
# is why two same-seed runs diverged at the very first optimiser step.
#
# Each dataset item derives its own RNG from (seed, index), so sampling is a
# function of the item alone -- invariant to worker count, batch order and
# scheduling, which process- or worker-level seeding cannot guarantee.
# ---------------------------------------------------------------------------
_ITEM_RNG = None


def _set_item_rng(seed):
    global _ITEM_RNG
    _ITEM_RNG = np.random.RandomState(seed % (2 ** 31 - 1))
    random.seed(seed % (2 ** 31 - 1))
    np.random.seed(seed % (2 ** 31 - 1))


def _next_transform_seed():
    """Deterministic seed for a MONAI transform, or None if unseeded mode."""
    if _ITEM_RNG is None:
        return None
    return int(_ITEM_RNG.randint(0, 2 ** 31 - 1))

try:
    from . import config
    from . import image_utils
except ImportError:
    import config
    import image_utils

# DataModule


# def collate_fn(datas):
#     '''
#     data is a list [[patch_list,positive_pairs_ls],[patch_list,positive_pairs_ls]]
#     return ([list_of_achors(1stpatch): n achors],[list_of_2nd_patch): n achors]], positive_pairs_ls_ls)
#     '''
#     # print(len(datas))

#     if len(datas[0]) == 2:
#         # print("In Not reconstruction")
#         anchor_ls = []
#         second_patch_ls = []
#         positive_pairs_ls_ls = []
#         # print(f"started appending the data")
#         for data in datas:
#             # print(f"started anchor_ls.append ")
#             anchor_ls.append(data[0][0].unsqueeze(dim=0))
#             # print(f"started second_patch_ls.append ")
#             second_patch_ls.append(data[0][1].unsqueeze(dim=0))
#             # print(data[1])
#             # print(f"started process_one_sample_pos_cord_list ")
#             positive_pairs_ls_ls.append(
#                 process_one_sample_pos_cord_list(data[1]))
#         # print(f"started anchor_torch concatenation")
#         anchor_torch = torch.cat(anchor_ls, dim=0)
#         # print(f"started second_patch_torch concatenation")
#         second_patch_torch = torch.cat(second_patch_ls, dim=0)
#         # print(f"started positive_pairs_ls_ls concatenation")
#         positive_pairs_ls_ls = torch.cat(positive_pairs_ls_ls, dim=0)
#         return ([anchor_torch, second_patch_torch], positive_pairs_ls_ls)

#     else:
#         print("In  reconstruction")

#         anchor_ls = []
#         second_patch_ls = []
#         positive_pairs_ls_ls = []
#         reconstruction_ls = []
#         reconstruction_mask_ls = []
#         for data in datas:
#             anchor_ls.append(data[0][0].unsqueeze(dim=0))
#             second_patch_ls.append(data[0][1].unsqueeze(dim=0))

#             print((data[2][0]).shape)
#             reconstruction_ls.append(data[2][0])
#             print((data[2][1]).shape)
#             reconstruction_mask_ls.append(data[2][1])

#             # print(data[1])
#             positive_pairs_ls_ls.append(
#                 process_one_sample_pos_cord_list(data[1]))

#         anchor_torch = torch.cat(anchor_ls, dim=0)
#         second_patch_torch = torch.cat(second_patch_ls, dim=0)
#         positive_pairs_ls_ls = torch.cat(positive_pairs_ls_ls, dim=0)
#         reconstruction_ls = torch.cat(reconstruction_ls, dim=0)
#         reconstruction_mask_ls = torch.cat(reconstruction_mask_ls, dim=0)

#         return ([anchor_torch, second_patch_torch], positive_pairs_ls_ls, [reconstruction_ls, reconstruction_mask_ls])


def _seed_worker(worker_id):
    """Give every DataLoader worker a deterministic, distinct RNG stream.

    Without this each worker re-seeds from OS entropy, so patch sampling and
    augmentation differ run to run even when the parent process is seeded.
    """
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class VoxelEmbedDataModule(pl.LightningDataModule):

    def __init__(self, args):
        super().__init__()

        # REPRODUCIBILITY: seed threaded through from the training config.
        self.seed = int(args.get('seed', 0)) if hasattr(args, 'get') else 0

        self.batch_size = args['batch_size']

        self.patch_size = args['patch_size']
        self.n_pos_voxel = args['n_pos_voxel']
        self.prob_overlapped_pairs = args['prob_overlapped_pairs']
        self.fg_pct = args['fg_pct']

        self.train_img_dir = args['train_img_dir']
        self.val_img_dir = args['val_img_dir']
        self.n_samples = args['n_samples']
        self.do_reconst = False
        self.recon_spatial_size = (8, 8, 8)

    def get_data_loader(self):

        # t1_raw_img_path = os.path.join(root_path,"data","raw_data","ae_train")
        # t1_raw_img_path_val = os.path.join(root_path,"data","raw_data","ae_val")
        # t1_raw_img_path_test = os.path.join(root_path,"data","raw_data","T1_orig","test")

        t1_raw_img_path = self.train_img_dir
        t1_raw_img_path_val = self.val_img_dir

        # REPRODUCIBILITY: os.listdir returns filesystem order, which is not
        # stable across runs or machines. Sort first, then shuffle with a
        # dedicated seeded RNG so the ordering is a function of the seed alone.
        img_paths = [os.path.join(t1_raw_img_path, idx)
                     for idx in sorted(os.listdir(t1_raw_img_path))]
        random.Random(self.seed).shuffle(img_paths)
        # print(img_paths)
        if self.n_samples != -1:
            img_paths = img_paths[:self.n_samples]

        self.n_samples = len(img_paths)
        print(f"Total Number of MRI images are {self.n_samples}")

        # prob_overlapped_pairs=0.95
        # fg_pct=0.8
        self.dataset_train = patch_dataset_from_file(img_paths, self.patch_size, self.n_pos_voxel, file_type="T1_brain.nii.gz",
                                                     prob_overlapped_pairs=self.prob_overlapped_pairs,
                                                     fg_pct=self.fg_pct, do_reconst=self.do_reconst,
                                                     recon_spatial_size=self.recon_spatial_size,
                                                     seed=self.seed)

        # dataloader_iter = iter(dataloader)

        # validation data loader
        img_paths_val = [os.path.join(t1_raw_img_path_val, idx)
                         for idx in sorted(os.listdir(t1_raw_img_path_val))]
        # random.shuffle(img_paths)
        # print(img_paths)
        if self.n_samples != -1:
            img_paths_val = img_paths_val[:self.n_samples]

        n_samples_val = len(img_paths_val)
        print(f"Total Number of MRI images are validation {n_samples_val}")

        # prob_overlapped_pairs=0.95
        # fg_pct=0.8
        self.dataset_val = patch_dataset_from_file(img_paths_val, self.patch_size, self.n_pos_voxel, file_type="T1_brain.nii.gz",
                                                   prob_overlapped_pairs=self.prob_overlapped_pairs,
                                                   fg_pct=self.fg_pct,
                                                   seed=None if self.seed is None else self.seed + 7919)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.get_data_loader()

    def train_dataloader(self):
        # REPRODUCIBILITY: an explicit generator makes the shuffle a function of
        # the seed; worker_init_fn stops each worker re-seeding from entropy.
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(self.dataset_train, batch_size=self.batch_size,
                          shuffle=True, num_workers=5,
                          collate_fn=collate_fn,
                          persistent_workers=True,
                          generator=generator,
                          worker_init_fn=_seed_worker,
                          drop_last=True)

    def val_dataloader(self):
        # not shuffled, but the workers still sample patches and augmentations
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          shuffle=False, num_workers=5,
                          persistent_workers=True,
                          worker_init_fn=_seed_worker,
                          collate_fn=collate_fn, drop_last=True)


######################## Data Processing ###########################


########################### Start from here for Foreground image pipeline##########


def ravel(x: NdarrayOrTensor):
    """`np.ravel` with equivalent implementation for torch.
    Args:
        x: array/tensor to ravel
    Returns:
        Return a contiguous flattened array/tensor.
    """
    if isinstance(x, torch.Tensor):
        if hasattr(torch, "ravel"):
            return x.ravel()
        return x.flatten().contiguous()
    return np.ravel(x)


def unravel_index(idx, shape):
    """`np.unravel_index` with equivalent implementation for torch.
    Args:
        idx: index to unravel
        b: shape of array/tensor
    Returns:
        Index unravelled for given shape
    """
    if isinstance(idx, torch.Tensor):
        coord = []
        for dim in reversed(shape):
            coord.insert(0, idx % dim)
            idx = floor_divide(idx, dim)
        return torch.stack(coord)
    return np.unravel_index(np.asarray(idx, dtype=int), shape)


def generate_pos_neg_label_crop_centers(
    spatial_size: Union[Sequence[int], int],
    num_samples: int,
    pos_ratio: float,
    label_spatial_shape: Sequence[int],
    fg_indices: NdarrayOrTensor,
    bg_indices: NdarrayOrTensor,
    rand_state: Optional[np.random.RandomState] = None,
) -> List[List[int]]:
    """
    Generate valid sample locations based on the label with option for specifying foreground ratio
    Valid: samples sitting entirely within image, expected input shape: [C, H, W, D] or [C, H, W]
    Args:
        spatial_size: spatial size of the ROIs to be sampled.
        num_samples: total sample centers to be generated.
        pos_ratio: ratio of total locations generated that have center being foreground.
        label_spatial_shape: spatial shape of the original label data to unravel selected centers.
        fg_indices: pre-computed foreground indices in 1 dimension.
        bg_indices: pre-computed background indices in 1 dimension.
        rand_state: numpy randomState object to align with other modules.
    Raises:
        ValueError: When the proposed roi is larger than the image.
        ValueError: When the foreground and background indices lengths are 0.
    """
    if rand_state is None:
        rand_state = np.random.random.__self__  # type: ignore

    centers = []
    fg_indices = np.asarray(fg_indices) if isinstance(
        fg_indices, Sequence) else fg_indices
    bg_indices = np.asarray(bg_indices) if isinstance(
        bg_indices, Sequence) else bg_indices
    if len(fg_indices) == 0 and len(bg_indices) == 0:
        raise ValueError("No sampling location available.")

    if len(fg_indices) == 0 or len(bg_indices) == 0:
        warnings.warn(
            f"N foreground {len(fg_indices)}, N  background {len(bg_indices)},"
            "unable to generate class balanced samples."
        )
        pos_ratio = 0 if fg_indices.size == 0 else 1

    for _ in range(num_samples):
        indices_to_use = fg_indices  # if rand_state.rand() < pos_ratio else bg_indices
        random_int = rand_state.randint(len(indices_to_use))
        idx = indices_to_use[random_int]
        center = unravel_index(idx, label_spatial_shape)
        # shift center to range of valid centers
        center_ori = list(center)
        centers.append(correct_crop_centers(
            center_ori, spatial_size, label_spatial_shape))

    return centers


class SpatialCrop(Transform):
    """
    General purpose cropper to produce sub-volume region of interest (ROI).
    If a dimension of the expected ROI size is bigger than the input image size, will not crop that dimension.
    So the cropped result may be smaller than the expected ROI, and the cropped results of several images may
    not have exactly the same shape.
    It can support to crop ND spatial (channel-first) data.

    The cropped region can be parameterised in various ways:
        - a list of slices for each spatial dimension (allows for use of -ve indexing and `None`)
        - a spatial center and size
        - the start and end coordinates of the ROI
    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        roi_center: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_size: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_start: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_end: Union[Sequence[int], NdarrayOrTensor, None] = None,
        roi_slices: Optional[Sequence[slice]] = None,
    ) -> None:
        """
        Args:
            roi_center: voxel coordinates for center of the crop ROI.
            roi_size: size of the crop ROI, if a dimension of ROI size is bigger than image size,
                will not crop that dimension of the image.
            roi_start: voxel coordinates for start of the crop ROI.
            roi_end: voxel coordinates for end of the crop ROI, if a coordinate is out of image,
                use the end coordinate of image.
            roi_slices: list of slices for each of the spatial dimensions.
        """
        roi_start_torch: torch.Tensor

        if roi_slices:
            if not all(s.step is None or s.step == 1 for s in roi_slices):
                raise ValueError(
                    "Only slice steps of 1/None are currently supported")
            self.slices = list(roi_slices)
        else:
            if roi_center is not None and roi_size is not None:
                roi_center = torch.as_tensor(roi_center, dtype=torch.int16)
                roi_size = torch.as_tensor(
                    roi_size, dtype=torch.int16, device=roi_center.device)
                self.roi_start_torch = maximum(  # type: ignore
                    roi_center - floor_divide(roi_size, 2),
                    torch.zeros_like(roi_center),
                )
                self.roi_end_torch = maximum(
                    self.roi_start_torch + roi_size, self.roi_start_torch)
            else:
                if roi_start is None or roi_end is None:
                    raise ValueError(
                        "Please specify either roi_center, roi_size or roi_start, roi_end.")
                self.roi_start_torch = torch.as_tensor(
                    roi_start, dtype=torch.int16)
                self.roi_start_torch = maximum(self.roi_start_torch, torch.zeros_like(
                    self.roi_start_torch))  # type: ignore
                self.roi_end_torch = maximum(torch.as_tensor(
                    roi_end, dtype=torch.int16), self.roi_start_torch)
            # convert to slices (accounting for 1d)
            if self.roi_start_torch.numel() == 1:
                self.slices = [
                    slice(int(self.roi_start_torch.item()), int(self.roi_end_torch.item()))]
            else:
                self.slices = [slice(int(s.item()), int(e.item())) for s, e in zip(
                    self.roi_start_torch, self.roi_end_torch)]

    def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
        """
        Apply the transform to `img`, assuming `img` is channel-first and
        slicing doesn't apply to the channel dim.
        """
        sd = min(len(self.slices), len(img.shape[1:]))  # spatial dims
        slices = [slice(None)] + self.slices[:sd]
        return img[tuple(slices)], (self.roi_start_torch, self.roi_end_torch)


class RandCropByPosNegLabel_custom(Randomizable, Transform):
    """
    Crop random fixed sized regions with the center being a foreground or background voxel
    based on the Pos Neg Ratio.
    And will return a list of arrays for all the cropped images.
    For example, crop two (3 x 3) arrays from (5 x 5) array with pos/neg=1::

        [[[0, 0, 0, 0, 0],
          [0, 1, 2, 1, 0],            [[0, 1, 2],     [[2, 1, 0],
          [0, 1, 3, 0, 0],     -->     [0, 1, 3],      [3, 0, 0],
          [0, 0, 0, 0, 0],             [0, 0, 0]]      [0, 0, 0]]
          [0, 0, 0, 0, 0]]]

    If a dimension of the expected spatial size is bigger than the input image size,
    will not crop that dimension. So the cropped result may be smaller than expected size, and the cropped
    results of several images may not have exactly same shape.

    Args:
        spatial_size: the spatial size of the crop region e.g. [224, 224, 128].
            if a dimension of ROI size is bigger than image size, will not crop that dimension of the image.
            if its components have non-positive values, the corresponding size of `label` will be used.
            for example: if the spatial size of input data is [40, 40, 40] and `spatial_size=[32, 64, -1]`,
            the spatial size of output data will be [32, 40, 40].
        label: the label image that is used for finding foreground/background, if None, must set at
            `self.__call__`.  Non-zero indicates foreground, zero indicates background.
        pos: used with `neg` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        neg: used with `pos` together to calculate the ratio ``pos / (pos + neg)`` for the probability
            to pick a foreground voxel as a center rather than a background voxel.
        num_samples: number of samples (crop regions) to take in each list.
        image: optional image data to help select valid area, can be same as `img` or another image array.
            if not None, use ``label == 0 & image > image_threshold`` to select the negative
            sample (background) center. So the crop center will only come from the valid image areas.
        image_threshold: if enabled `image`, use ``image > image_threshold`` to determine
            the valid image content areas.
        fg_indices: if provided pre-computed foreground indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.
        bg_indices: if provided pre-computed background indices of `label`, will ignore above `image` and
            `image_threshold`, and randomly select crop centers based on them, need to provide `fg_indices`
            and `bg_indices` together, expect to be 1 dim array of spatial indices after flattening.
            a typical usage is to call `FgBgToIndices` transform first and cache the results.

    Raises:
        ValueError: When ``pos`` or ``neg`` are negative.
        ValueError: When ``pos=0`` and ``neg=0``. Incompatible values.

    """

    backend = [TransformBackends.TORCH, TransformBackends.NUMPY]

    def __init__(
        self,
        spatial_size: Union[Sequence[int], int],
        label: Optional[NdarrayOrTensor] = None,
        pos: float = 1.0,
        neg: float = 1.0,
        num_samples: int = 1,
        image: Optional[NdarrayOrTensor] = None,
        image_threshold: float = 0.0,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
    ) -> None:
        self.spatial_size = ensure_tuple(spatial_size)
        self.label = label
        if pos < 0 or neg < 0:
            raise ValueError(
                f"pos and neg must be nonnegative, got pos={pos} neg={neg}.")
        if pos + neg == 0:
            raise ValueError("Incompatible values: pos=0 and neg=0.")
        self.pos_ratio = pos / (pos + neg)
        self.num_samples = num_samples
        self.image = image
        self.image_threshold = image_threshold
        self.centers: Optional[List[List[int]]] = None
        self.fg_indices = fg_indices
        self.bg_indices = bg_indices
        self.comp_fg_indices = None
        self.comp_bg_indices = None

    def randomize(
        self,
        label: NdarrayOrTensor,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
        image: Optional[NdarrayOrTensor] = None,
    ) -> None:
        self.spatial_size = fall_back_tuple(
            self.spatial_size, default=label.shape[1:])
        if fg_indices is None or bg_indices is None:
            if self.fg_indices is not None and self.bg_indices is not None:
                fg_indices_ = self.fg_indices
                bg_indices_ = self.bg_indices
            else:
                fg_indices_, bg_indices_ = map_binary_to_indices(
                    label, image, self.image_threshold)
                self.comp_fg_indices, self.comp_bg_indices = fg_indices_, bg_indices_
        else:
            fg_indices_ = fg_indices
            bg_indices_ = bg_indices
        self.centers = generate_pos_neg_label_crop_centers(
            self.spatial_size, self.num_samples, self.pos_ratio, label.shape[
                1:], fg_indices_, bg_indices_, self.R
        )

    def __call__(
        self,
        img: NdarrayOrTensor,
        label: Optional[NdarrayOrTensor] = None,
        image: Optional[NdarrayOrTensor] = None,
        fg_indices: Optional[NdarrayOrTensor] = None,
        bg_indices: Optional[NdarrayOrTensor] = None,
    ) -> List[NdarrayOrTensor]:
        """
        Args:
            img: input data to crop samples from based on the pos/neg ratio of `label` and `image`.
                Assumes `img` is a channel-first array.
            label: the label image that is used for finding foreground/background, if None, use `self.label`.
            image: optional image data to help select valid area, can be same as `img` or another image array.
                use ``label == 0 & image > image_threshold`` to select the negative sample(background) center.
                so the crop center will only exist on valid image area. if None, use `self.image`.
            fg_indices: foreground indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.
            bg_indices: background indices to randomly select crop centers,
                need to provide `fg_indices` and `bg_indices` together.

        """
        if label is None:
            label = self.label
        if label is None:
            raise ValueError("label should be provided.")
        if image is None:
            image = self.image

        self.randomize(label, fg_indices, bg_indices, image)
        results: List[NdarrayOrTensor] = []
        results_crop_lbels: List[NdarrayOrTensor] = []
        coords_ls = []
        if self.centers is not None:
            for center in self.centers:
                cropper = SpatialCrop(roi_center=tuple(
                    center), roi_size=self.spatial_size)
                img, coords = cropper(img)
                label_crop, _ = cropper(label)
                results.append(img)
                results_crop_lbels.append(label_crop)
                coords_ls.append(coords)

        return results, coords_ls, (self.comp_fg_indices), results_crop_lbels


def unravel_point(np_point, original_shape):
    '''
    It returns the value of flattened array which has the same shape of original_shape. For example, the index
    [10,12,21] of an array of size [12,15,27] would be 4395.
    '''
    if isinstance(np_point, torch.Tensor):
        np_point = np_point.numpy()

    # get multiplicative coefficient
    coeffs = np.asarray(return_coeff_to_unravel_point(original_shape))

    return (np_point*coeffs).sum(axis=-1)


def return_coeff_to_unravel_point(lists):
    '''
    it returns the coefficints  [y*z , z, 1] of this expression (a,b,c). a*(y*z) + (b*z) +(c*1)

    '''

    if isinstance(lists, torch.Tensor):
        lists = lists.numpy()

    elif isinstance(lists, (list, tuple)):
        lists = np.asarray(lists)

    ls = []

    def cons_multiply(list_num, ob):
        if len(list_num) == 1:
            ob.append(1)
        else:
            ob.append(np.prod(list_num[1:]))
            cons_multiply(list_num[1:], ob)

    cons_multiply(lists, ls)
    return ls


'''
d_point = unravel_point(np.array([10,12,21]),(12,15,27))
d_point # returns 4395

'''

# determine what percentage of foreground are selected in the sampled cube


def get_tot_fg_in_cube(endpoints, image_shape, fg_indices):
    '''
    Based on corner points (start and end points of the cube), and image_shaep, foreground point,
    it returns the index of fore ground points in cube.

    Arguments:
        endpoints : [(start cords),(end coords) ]
        image_shape = (xdim,ydim,zdim)
        fg_indices = 1-d array (numpy) containing the flatten index of foreground voxels

    Returns:
        tot_fg_in_cube : 1-d array of foreground voxels in the patch
    '''

    x_s, y_s, z_s = endpoints[0][0]  # start point
    x_e, y_e, z_e = endpoints[0][1]  # end point

    # what are the coords of points inside the cube  in original shape, then 1-d)

    dummy = torch.zeros(image_shape)
    dummy[x_s:x_e, y_s:y_e, z_s:z_e] = 1
    mask = torch.nonzero(dummy)

    # what are  the 1-d coords of those points
    # [unravel_point(p,image_shape) for p in mask]
    mask = unravel_point(mask, image_shape)

    # what are the overlaps between those points and fg
    tot_fg_in_cube = np.intersect1d(np.asarray(
        mask), fg_indices, return_indices=False).astype(int)

    return tot_fg_in_cube


def draw(p_pos):
    '''
    Draw Either 0 or 1 based on the probability of drawing a 1

    '''
    from numpy.random import choice
    p_pos = p_pos
    list_of_candidates = [0, 1]
    number_of_items_to_pick = 1
    probability_distributio = [1-p_pos, p_pos]
    draw = choice(list_of_candidates, number_of_items_to_pick,
                  p=probability_distributio)
    return draw
# draw(0.1)


def draw_paired_overlapped_patch(endpoints, tot_fg_in_cube, fg_indices, image_shape, spatial_size, mri_image, tot_point_cube, brain_mask):
    '''
    endpoints: the endpoints [(s_x,e_x),(s_y,e_y), (s_z,e_z)] of first patch

    tot_fg_in_cube: absolute number of foreground points in the first patch we have
    fg_indices: indices of the foreground voxels
    image_shape: original MRI shape
    spatial_size: shape of patch
    mri_image: original MRI image (c,x,y,z)
    tot_point_cube: total number of voxels in the patch.
    '''

    # x_s,y_s,z_s = endpoints[0][0] # start point
    indices_to_use = tot_fg_in_cube
    random_int = np.random.randint(len(indices_to_use))
    idx = indices_to_use[random_int]
    center = unravel_index(idx, image_shape)
    # shift center to range of valid centers
    center_ori = list(center)
    center_final = correct_crop_centers(center_ori, spatial_size, image_shape)

    cropper = SpatialCrop(roi_center=tuple(
        center_final), roi_size=spatial_size)
    img_pair, endpoints_pair = cropper(mri_image)
    crop_pair, _ = cropper(brain_mask)

    tot_fg_in_cube_pair = get_tot_fg_in_cube(
        [endpoints_pair], image_shape, fg_indices)
    fg_perct_pair = len(tot_fg_in_cube_pair)/tot_point_cube

    return img_pair, endpoints_pair, tot_fg_in_cube_pair, fg_perct_pair, crop_pair


def draw_valid_pair(brain, brain_mask, prob_overlapped_pairs, spatial_size, per_fg=0.75):
    '''

    Draw two valid random cube/patch. Both of them will be of foreground and atleast prob_overlapped_pairs of the points
    in the patch will be from foreground.


    Arguments:
        brain: Original Image .torch.Tensor : (channel, x,y,z)
        brain_masktorch.Tensor : Mask containing foreground and background. (channel, x,y,z)
        prob_overlapped_pairs : fraction. minimum percentage of points which should be from foreground
        spatial_size: tuple. spatial shape of the patch. (x,y,z)
    Returns:
        [random_Crops,random_Crops_pair]: torch.tensor. Two random patch.
        [endpoints,endpoints_pair] : coordinates of endpoints of each of the patch. [[(start_point_p1:tensor),(end_point_p1:tensor))],[((start_point_p2:tensor),(end_point_p2:tensor)]) ]
        [tot_fg_in_cube,tot_fg_in_cube_pair] : list of 1-D tensor. points (1-D mapped) which are from foreground(in original space) of each of the patch
    randcrp_cls=  RandCropByPosNegLabel_custom(spatial_size, pos=pos, neg=neg, num_samples=1)
    random_Crops,endpoints,fg_indices = randcrp_cls(brain, label = brain_mask)

    '''
    # print("draw_valid_pair")
    pos = 100000
    neg = .000001
    num_samples = 1
    randcrp_cls = RandCropByPosNegLabel_custom(
        spatial_size, pos=pos, neg=neg, num_samples=1)
    # REPRODUCIBILITY: reach MONAI's entropy-seeded RandomState
    _tseed = _next_transform_seed()
    if _tseed is not None:
        randcrp_cls.set_random_state(seed=_tseed)

    fg_perct = 0.0
    tot_point_cube = torch.prod(torch.tensor(spatial_size)).item()
    image_shape = brain.shape[1:]
    i = 0
    p_pos = prob_overlapped_pairs  # .95
    # get the first sample
    # print("Drawing the first patch")
    while fg_perct < per_fg:

        random_Crops, endpoints, fg_indices, fgmask_Crops = randcrp_cls(
            brain, label=brain_mask)

        tot_fg_in_cube = get_tot_fg_in_cube(endpoints, image_shape, fg_indices)
        fg_perct = len(tot_fg_in_cube)/tot_point_cube
        # print(f"fg_perct: {fg_perct}")
    # now select the second patch which should have overlap with the first one more than 95% time
    # print(f"Done drawing first patch")
    if draw(p_pos) == 1:  # overlapping drawing
        # print("Now taking an overlapped patch")

        fg_perct_pair = 0.0
        while fg_perct_pair < per_fg:
            random_Crops_pair, endpoints_pair, tot_fg_in_cube_pair, fg_perct_pair, fgmask_random_Crops_pair = draw_paired_overlapped_patch(endpoints, tot_fg_in_cube, fg_indices,
                                                                                                                                           image_shape, spatial_size, brain,
                                                                                                                                           tot_point_cube, brain_mask)
        # print("Done taking the second overlapped patch")
    else:  # draw randomly
        # print("Now taking an Non patch")
        fg_perct = 0.0
        while fg_perct < per_fg:

            random_Crops_pair, endpoints_pair, fg_indices_pair = randcrp_cls(
                brain, label=brain_mask)
            tot_fg_in_cube_pair = get_tot_fg_in_cube(
                endpoints_pair, image_shape, fg_indices_pair)
            fg_perct = len(tot_fg_in_cube)/tot_point_cube
            # print(f"fg_perct: {fg_perct}")
        # print("Done taking the second None patch")

    if isinstance(random_Crops_pair, list):
        random_Crops_pair = random_Crops_pair[0]
        fgmask_random_Crops_pair = fgmask_random_Crops_pair[0]

    if isinstance(random_Crops, list):
        random_Crops = random_Crops[0]
        fgmask_Crops = fgmask_Crops[0]
    return [random_Crops, random_Crops_pair], [endpoints, [endpoints_pair]], [tot_fg_in_cube, tot_fg_in_cube_pair, fg_indices], [fgmask_Crops, fgmask_random_Crops_pair]


'''

brain_mask = nib.load("T1_brain_mask.nii.gz").get_fdata().reshape(1,170, 243, 202)
brain = nib.load("T1_brain.nii.gz").get_fdata().reshape(1,170, 243, 202)
prob_overlapped_pairs = 0.95
spatial_size = (6,6,6)

[random_Crops,random_Crops_pair],[endpoints,endpoints_pair],[tot_fg_in_cube,tot_fg_in_cube_pair] = draw_valid_pair(brain,brain_mask, prob_overlapped_pairs,spatial_size)
#random_Crops,random_Crops_pair
#endpoints,endpoints_pair
#tot_fg_in_cube,tot_fg_in_cube_pair
'''


def get_positive_voxel_coords(coord_list, fg_list, n_pos_pairs, image_shape):
    '''
    Choose random positive voxel coordinate in anchor and paired patch
    Argument:
        coord_list: list of tensor. start and end coordinates of two patches
        eg.
        ([(tensor([127, 176,  99], dtype=torch.int16),
           tensor([133, 182, 105], dtype=torch.int16))], #start and end of first patch

         [(tensor([127, 176,  99], dtype=torch.int16),
           tensor([133, 182, 105], dtype=torch.int16))] #start and end of second patch

           )
        fg_list: list of 1-D array(numpy). [1-D array of positive points in first patch, 1-D array of positive points in second patch]
        n_pos_pairs = int, how many positive voxels we want
        image_shape: (x,y,z) . Original image shape


    Returns:
    positive_pairs_ls : list of numpy array. [first voxel: array(2*3, first row ancho, 2nd row pair) ,2nd voxel: array(2*3, first row ancho, 2nd row pair),....]
    '''

    # check whether they intersect
    # print(f"coordlist:{coord_list}")
    tot_fg_in_cube, tot_fg_in_cube_pair = fg_list
    intersect = np.intersect1d(
        tot_fg_in_cube, tot_fg_in_cube_pair, return_indices=False).astype(int)
    if intersect.size > 0:
        # print("Did  overlap")
        # intersect

        positive_pairs_ls = []
        for pos in range(n_pos_pairs):

            # choose 1 forground voxel

            pos_voxel = unravel_index(np.random.choice(intersect), image_shape)
            # print(pos_voxel)

            # get the patch for reconstruction

            # get coordinates in space of first patch
            p1_pos_cord = torch.tensor(pos_voxel) - coord_list[0][0][0]
            p1_pos_cord = p1_pos_cord.numpy()

            if isinstance(coord_list[1][0][0], tuple):
                p2_orig = coord_list[1][0][0][0]
            else:
                p2_orig = coord_list[1][0][0]
            # print(f"p2_orig: {p2_orig}")
            p2_pos_cord = torch.tensor(pos_voxel) - p2_orig
            # p2_pos_cord = torch.tensor(pos_voxel) - coord_list[1][0][0]
            p2_pos_cord = p2_pos_cord.numpy()

            pos = np.concatenate(
                (p1_pos_cord, p2_pos_cord), axis=0).reshape(2, 3)

            positive_pairs_ls.append(pos)
        # print(positive_pairs_ls)
        return positive_pairs_ls

    else:  # did not overlap, chosse n_pos_pairs from first patch
        # print("Did not overlap")
        positive_pairs_ls = []

        for pos in range(n_pos_pairs):

            # choose 1 forground voxel

            pos_voxel = unravel_index(
                np.random.choice(tot_fg_in_cube), image_shape)
            # print(pos_voxel)
            # get coordinates in space of first patch
            p1_pos_cord = torch.tensor(pos_voxel) - coord_list[0][0][0]
            p1_pos_cord = p1_pos_cord.numpy()

            pos = np.concatenate(
                (p1_pos_cord, p1_pos_cord), axis=0).reshape(2, 3)

            positive_pairs_ls.append(pos)

        return positive_pairs_ls


'''        
brain_mask = nib.load("T1_brain_mask.nii.gz").get_fdata().reshape(1,170, 243, 202)
brain = nib.load("T1_brain.nii.gz").get_fdata().reshape(1,170, 243, 202)
prob_overlapped_pairs = 0.95
spatial_size = (6,6,6)

[random_Crops,random_Crops_pair],[endpoints,endpoints_pair],[tot_fg_in_cube,tot_fg_in_cube_pair] = draw_valid_pair(brain,brain_mask, prob_overlapped_pairs,spatial_size)

coord_list = [endpoints,endpoints_pair]
fg_list = [tot_fg_in_cube,tot_fg_in_cube_pair]
n_pos_pairs = 10
image_shape = brain.shape[1:]
positive_pairs_ls = get_positive_voxel_coords(coord_list,fg_list,n_pos_pairs,image_shape)
positive_pairs_ls
'''


def get_all_coordinates_inside_a_cube(boundary_points, image_shape):
    star_idx, end_idx = boundary_points
    s_x, s_y, s_z = star_idx
    e_x, e_y, e_z = end_idx
    d = torch.arange(torch.prod(torch.tensor(image_shape))
                     ).reshape(image_shape)
    indices = d[s_x:e_x, s_y:e_y, s_z:e_z].flatten()

    return indices, star_idx.numpy(), end_idx.numpy()


def get_mask_reconstruction_cube(boundary_points, image_shape, fg_indices, recon_spatial_size):
    '''
    Given the boundary points of cube around the positive voxel, it will give the foreground maks for 
    this cube.

    Arguments:
        boundary_points: list of arrary: [[x,y,z coordinate of top left corner],[x,y,z coordinate of bottom right corner]]
        image_shape: Original MRI shape.tuple(x,y,x)
        fg_indices: 1-d array. 1-d flatten indices of foreground voxels in MRI.
        recon_spatial_size: Shape of reconstruction cube. tuple(x,y,z)

    return :
        d: mask. shape= recon_spatial_size. foreground voxels value is 1. 3-d np array
    '''

    # get the coordinates sampled cube. in 1-d format
    all_indices, star_idx, end_idx = get_all_coordinates_inside_a_cube(
        boundary_points, image_shape)

    # get all the foreground flatten voxels indices within the cube
    fg_recon_patch = np.intersect1d(
        all_indices, fg_indices, return_indices=False).astype(int)

    # get the indices of foreground voxels in cube in terms of original shape
    fg_recon_patch_unraveled = np.unravel_index(fg_recon_patch, image_shape)
    fg_recon_patch_unraveled = np.vstack(fg_recon_patch_unraveled).transpose()

    # compute the foreground coordinates with respect to top left corner of the cube
    s = (fg_recon_patch_unraveled-star_idx)

    # create the mask
    d = np.zeros(recon_spatial_size)
    d[s] = 1

    return d


def get_reconstruction_cube(img, pos_voxel, recon_spatial_size, fg_indices, image_shape):
    '''Return the cube sample around the positive voxel. 

        Arguments:
            img: torch img. Shape (c,x,y,z)
            pos_voxel: np array. I-d array (3,). x,y,z coordinate of the chosen positive voxel
            recon_spatial_size: shape of reconstruction cube. tuple (x,y,z)
            fg_indices: 1-d flatten indices of foreground voxels in MRI.
            image_shape: shape of original image. tuple (x,y,z)
        Returns:
            img: torch.tensor. shape (c,*recon_spatial_size) : 4D
            mask: torch.tensor. shape (*recon_spatial_size):3D

    '''
    center_ori = list(pos_voxel)
    center_final = correct_crop_centers(
        center_ori, recon_spatial_size, image_shape)

    cropper = SpatialCrop(roi_center=tuple(center_final),
                          roi_size=recon_spatial_size)
    recon, recon_endpoint = cropper(img)

    # get the mask for the cube
    mask = get_mask_reconstruction_cube(
        recon_endpoint, image_shape, fg_indices, recon_spatial_size)

    return recon.unsqueeze(dim=0), torch.from_numpy(mask).unsqueeze(dim=0)


def get_positive_voxel_coords_with_reconstruction(coord_list, fg_list, n_pos_pairs, image_shape,

                                                  img,
                                                  fg_indices,
                                                  recon_spatial_size=(8, 8, 8)):
    '''
    Choose random positive voxel coordinate in anchor and paired patch, also return patch centered on positive voxel for reconstruction.

    Argument:
        coord_list: list of tensor. start and end coordinates of two patches
        eg.
        ([(tensor([127, 176,  99], dtype=torch.int16),
           tensor([133, 182, 105], dtype=torch.int16))], #start and end of first patch

         [(tensor([127, 176,  99], dtype=torch.int16),
           tensor([133, 182, 105], dtype=torch.int16))] #start and end of second patch

           )
        fg_list: list of 1-D array(numpy). [1-D array of positive points in first patch, 1-D array of positive points in second patch]
        n_pos_pairs = int, how many positive voxels we want
        image_shape: (x,y,z) . Original image shape


    Returns:
    positive_pairs_ls : list of numpy array. [first voxel: array(2*3, first row ancho, 2nd row pair) ,2nd voxel: array(2*3, first row ancho, 2nd row pair),....]
    '''

    # check whether they intersect
    # print(f"coordlist:{coord_list}")
    tot_fg_in_cube, tot_fg_in_cube_pair = fg_list
    intersect = np.intersect1d(
        tot_fg_in_cube, tot_fg_in_cube_pair, return_indices=False).astype(int)
    if intersect.size > 0:
        # print("Did  overlap")
        # intersect

        positive_pairs_ls = []
        recons_patch_ls = []
        recons_mask_ls = []
        for pos in range(n_pos_pairs):

            # choose 1 forground voxel

            pos_voxel = unravel_index(np.random.choice(intersect), image_shape)
            # print(pos_voxel)

            # get the patch for reconstruction
            rec_cube, rec_mask = get_reconstruction_cube(
                img, pos_voxel, recon_spatial_size, fg_indices, image_shape)

            # get coordinates in space of first patch
            p1_pos_cord = torch.tensor(pos_voxel) - coord_list[0][0][0]
            p1_pos_cord = p1_pos_cord.numpy()

            if isinstance(coord_list[1][0][0], tuple):
                p2_orig = coord_list[1][0][0][0]
            else:
                p2_orig = coord_list[1][0][0]
            # print(f"p2_orig: {p2_orig}")
            p2_pos_cord = torch.tensor(pos_voxel) - p2_orig
            # p2_pos_cord = torch.tensor(pos_voxel) - coord_list[1][0][0]
            p2_pos_cord = p2_pos_cord.numpy()

            pos = np.concatenate(
                (p1_pos_cord, p2_pos_cord), axis=0).reshape(2, 3)

            positive_pairs_ls.append(pos)

            recons_patch_ls.append(rec_cube)

            recons_mask_ls.append(rec_mask)

        recons_patch_ls = torch.concat(recons_patch_ls, axis=0)
        recons_mask_ls = torch.concat(recons_mask_ls, axis=0)
        # print(positive_pairs_ls)
        return positive_pairs_ls, recons_patch_ls, recons_mask_ls

    else:  # did not overlap, chosse n_pos_pairs from first patch
        # print("Did not overlap")
        positive_pairs_ls = []

        for pos in range(n_pos_pairs):

            # choose 1 forground voxel

            pos_voxel = unravel_index(
                np.random.choice(tot_fg_in_cube), image_shape)
            # print(pos_voxel)
            # get coordinates in space of first patch
            p1_pos_cord = torch.tensor(pos_voxel) - coord_list[0][0][0]
            p1_pos_cord = p1_pos_cord.numpy()

            pos = np.concatenate(
                (p1_pos_cord, p1_pos_cord), axis=0).reshape(2, 3)

            positive_pairs_ls.append(pos)

        return positive_pairs_ls


def process_one_sample(img, img_mask, n_pos_voxels, spatial_size,
                       prob_overlapped_pairs=0.80, per_fg=0.80, do_reconst=False,
                       recon_spatial_size=(8, 8, 8)):
    '''
    Return two valid patches and the coordinates of n_pos_pixels
    Arguments:
        img: np.array. (c,x,y,z)
        img_mask : np.array. (c,x,y,z)
        n_pos_voxels: int
        spatial_size: tuple.(x,y,z)
        prob_overlapped_pairs: float. the probability of sampling two overlapping patches
        per_fg: float. minimum percentage of foreground voxels in a patch to be accepted.

    Returns:
        torch.tensor : [random_Crops,random_Crops_pair]
        positive_pairs_ls : list: length of this list is n_pos_voxel : [torch.tensor_shape_(2,3), torch.tensor_shape_(2,3)]
    '''
    # print(f"started draw_valid_pair")
    ([random_Crops, random_Crops_pair], [endpoints, endpoints_pair],

     [tot_fg_in_cube, tot_fg_in_cube_pair, fg_indices], [fgmask_random_Crops, fgmask_random_Crops_pair]) = draw_valid_pair(brain=img, brain_mask=img_mask,
                                                                                                                           prob_overlapped_pairs=prob_overlapped_pairs,
                                                                                                                           spatial_size=spatial_size,
                                                                                                                           per_fg=per_fg)
    # print(f"Ended draw_valid_pair")

    coord_list = [endpoints, endpoints_pair]
    fg_list = [tot_fg_in_cube, tot_fg_in_cube_pair]

    image_shape = img.shape[1:]

    if do_reconst is not True:
        positive_pairs_ls = get_positive_voxel_coords(
            coord_list, fg_list, n_pos_voxels, image_shape)

        return [random_Crops, random_Crops_pair], [positive_pairs_ls], [fgmask_random_Crops, fgmask_random_Crops_pair]

    else:

        positive_pairs_ls, recons_patch_ls, recons_fg_ls = get_positive_voxel_coords_with_reconstruction(coord_list, fg_list, n_pos_voxels, image_shape,

                                                                                                         img,
                                                                                                         fg_indices,
                                                                                                         recon_spatial_size=recon_spatial_size)

        return [random_Crops, random_Crops_pair], positive_pairs_ls, [recons_patch_ls, recons_fg_ls]


''' 
img = nib.load("T1_brain_mask.nii.gz").get_fdata().reshape(1,170, 243, 202)
img_mask = nib.load("T1_brain.nii.gz").get_fdata().reshape(1,170, 243, 202)
prob_overlapped_pairs = 0.95
n_pos_voxels = 10
spatial_size = (6,6,6)

[random_Crops,random_Crops_pair],positive_pairs_ls = process_one_sample(img,img_mask,n_pos_voxels,spatial_size, prob_overlapped_pairs = 0.95)
'''


######################### DataLoader ###################################


def process_one_sample_pos_cord_list(pos_cord_ls):
    '''
    It will return a (n_pos_pixel,4) tensor. [:,:2] will be from positive voxels from ist patch, [:,2:] from the second patch

    Argument:
    pos_cord_ls: either length 2, if no overlapping, or length 1 in overlapping.
    if not overlapped, then each index has n_pos//2 number of 2*3 array.
    if overlapped, then there will be n_pos number of 2*2 array

    '''

    # Flatten to a (2*n_pos, 3) coordinate list -- ordered
    # [p0_v1, p0_v2, p1_v1, p1_v2, ...] -- then fold the two views of each
    # positive together. This is agnostic to whether the caller passes one
    # stacked (n_pos, 2, 3) array, one array per pair, or the two-element
    # non-overlapping layout.
    #
    # Getting this grouping wrong is silent but fatal: pairing view 1 of one
    # positive with view 1 of another leaves the local InfoNCE positive an
    # unrelated voxel, so that term sits at chance -- ln(1 + n_neg) -- and the
    # fine-scale branch collapses to a constant embedding. The instance term is
    # unaffected (it pools whole patches), and the coarse-scale term only partly
    # so, because its //16 downsampling often maps a mis-paired coordinate into
    # the same cell. A quick check: the raw image intensities at correctly
    # paired coordinates must be identical, since both views are crops of the
    # same volume.
    flat = np.concatenate(
        [np.asarray(p).reshape(-1, 3) for p in pos_cord_ls], axis=0)
    conc = flat.reshape(-1, 2, 3).reshape(-1, 6)   # [x1,y1,z1,x2,y2,z2]

    return torch.from_numpy(np.ascontiguousarray(conc))


def collate_fn(datas):
    '''
    data is a list [[patch_list,positive_pairs_ls],[patch_list,positive_pairs_ls]]
    return ([list_of_achors(1stpatch): n achors],[list_of_2nd_patch): n achors]], positive_pairs_ls_ls)
    '''
    anchor_ls = []
    second_patch_ls = []
    positive_pairs_ls_ls = []
    first_fg_mask_ls = []
    secnd_fg_mask_ls = []

    for data in datas:
        anchor_ls.append(data[0][0].unsqueeze(dim=0))
        second_patch_ls.append(data[0][1].unsqueeze(dim=0))
        positive_pairs_ls_ls.append(process_one_sample_pos_cord_list(data[1]))
        first_fg_mask_ls.append(data[2][0].unsqueeze(dim=0))
        secnd_fg_mask_ls.append(data[2][1].unsqueeze(dim=0))
    anchor_torch = torch.cat(anchor_ls, dim=0)
    second_patch_torch = torch.cat(second_patch_ls, dim=0)
    positive_pairs_ls_ls = torch.cat(positive_pairs_ls_ls, dim=0)
    first_fg_masks = torch.cat(first_fg_mask_ls, dim=0)
    secnd_fg_masks = torch.cat(secnd_fg_mask_ls, dim=0)
    return ([anchor_torch, second_patch_torch], positive_pairs_ls_ls, [first_fg_masks, secnd_fg_masks])


class patch_dataset_from_file(Dataset):

    def __init__(self, imgs_paths, patch_size, n_pos_pixels, file_type,
                 prob_overlapped_pairs=1.0, fg_pct=0.8, do_reconst=False,
                 recon_spatial_size=(8, 8, 8), seed=None
                 ):
        super(patch_dataset_from_file, self).__init__()
        # seed=None keeps the original unseeded behaviour
        self.seed = seed
        self.imgs_paths = imgs_paths
        self.patch_size = patch_size
        self.n_pos_pixels = n_pos_pixels
        self.file_type = file_type
        self.prob_overlapped_pairs = prob_overlapped_pairs
        self.fg_pct = fg_pct
        self.do_reconst = do_reconst
        self.recon_spatial_size = recon_spatial_size

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):
        """

        returns:
            if no reconstruction is needed:

                patch_list: [patch_1, patch_2]. each of these is :5D torch.tensor (n_pos_pixels, c,x,y,z)
                positive_pairs_ls: list of 2-D tensors (n_pos_pixels,,3,2). each positive voxel's and its pair's coordinate.
            if reconstruction was true:
                 [random_Crops,random_Crops_pair],[positive_pairs_ls]: same as patch_list & positive_pairs_ls
                 recons_patch_ls: n_pos_pixels,1,*(recon_spatial_size). 5 D tensor. the cube around each positive voxel
                 recons_mask_ls: (n_pos_pixels,1,*(recon_spatial_size)). 4D tensor. the background, foreground mask for each positive voxel's cube. foregrounds are 1.

        """

        # REPRODUCIBILITY: per-item RNG derived from (seed, idx)
        if self.seed is not None:
            _set_item_rng(self.seed * 1000003 + idx)

        img_path = self.imgs_paths[idx]
        # print(img_path)
        # print(f"in the dataset img shape is :{img.shape}")
        # patch_list : list of tensor ; positive_pairs_ls : list of tensor
        img, img_mask = get_torch_from_single_nifti(nifti_dir=img_path, file_type=self.file_type,
                                                    n_sample=1, save_numpy=False,
                                                    save_numpy_dir=None)
        # img = img.numpy()
        # img_mask = img_mask.numpy()

        output = process_one_sample(img=img, img_mask=img_mask,
                                    n_pos_voxels=self.n_pos_pixels,
                                    spatial_size=self.patch_size,
                                    prob_overlapped_pairs=self.prob_overlapped_pairs,
                                    per_fg=self.fg_pct,
                                    do_reconst=self.do_reconst,
                                    recon_spatial_size=self.recon_spatial_size)

        if self.do_reconst is not True:
            # print(f"NOT In reconstruction")
            patch_list, positive_pairs_ls, fgmasks = output
            return [patch_list, positive_pairs_ls, fgmasks]
        else:
            print(f"In reconstruction")
            [random_Crops, random_Crops_pair], positive_pairs_ls, [
                recons_patch_ls, recons_mask_ls] = output
            return ([random_Crops, random_Crops_pair], positive_pairs_ls, [recons_patch_ls, recons_mask_ls])


############################## Image Extraction##############################
# def extract_t1_brain(number_t1, save_dir,file_type = "T1/T1_brain_to_MNI.nii.gz",ukb_dir = "<path/to>/ukb_t1_source/",**kwargs):
#     '''
#     Extract number_t1 of T1 brain from <path/to>/ukb_t1_source/ to save_dir. Individual
#     folder is identified patientid_20252
#     ukb_dir from 122: <path/to>/ukb_t1_source/

#     '''

#     if "file_list" in kwargs:

#         # crate image dict
#         img_dir_1 = "<path/to>/ukb_t1_source"
#         img_dir_2 = "<path/to>/ukb_t1t2"


#         dir1_dict = {img:img_dir_1 for img in list(filter(lambda x: "20252" in x ,  os.listdir(img_dir_1)))}
#         dir2_dict = {img:img_dir_2 for img in list(filter(lambda x: "20252" in x , os.listdir(img_dir_2)))}

#         img_dict = {**dir1_dict, **dir2_dict}


#         t1_files = [f"{i}.zip" for i in kwargs['file_list']]
#         print(f"Total files: {len(t1_files)},{t1_files[0]}")

#         t1_files_full_path = [os.path.join(img_dict[f],f) for f in t1_files]

#     else:
#         files = os.listdir(ukb_dir)
#         t1_files = [f for f in files if f.endswith("zip") and f.split("_")[1] == "20252"]

#         #randomly permute and select
#         random.shuffle(t1_files)
#         t1_files = t1_files[:number_t1]


#         t1_files_full_path = [os.path.join(ukb_dir,f) for f in t1_files]

#     print(f"we are extracting {file_type} for total {len(t1_files_full_path)} samples")
#     for file in t1_files_full_path:

#         with ZipFile(file) as zipobj:

#             patient_id = "_".join((file.split("/")[-1]).split("_")[:2])
#             listOfFileNames = zipobj.namelist()

#             for file in listOfFileNames:
#                 if file in [file_type,"T1/T1_brain_mask.nii.gz"]:
#                     print(file)
#                     dir_path = os.path.join(save_dir,patient_id,file.split("/")[1])
#                     print(dir_path)
#                     zipobj.extract(file, dir_path)
#             print()
#     print(f"Extraction of {file_type} is completed")


def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result


def get_torch_from_single_nifti(nifti_dir,  file_type="T1_brain_to_MNI.nii.gz", n_sample=1, save_numpy=False, save_numpy_dir=None,
                                need_mask_extraction=True):

    result = find_all(file_type, nifti_dir)

    result = result if n_sample is None else result[:n_sample]

    n1_img = nib.load(result[0]).get_fdata()

    n1_img = torch.from_numpy(n1_img).to(torch.float)

    max_intensity = torch.max(n1_img)
    min_intensity = torch.min(n1_img)
    mri_data_rescaled = (n1_img - min_intensity) / \
        (max_intensity - min_intensity)
    n1_img = mri_data_rescaled.unsqueeze(0)
    # n1_img = n1_img/n1_img.max()
    # n1_img = (n1_img / 255.0) * 2 - 1

    if need_mask_extraction:
        file_type = "T1_brain_mask.nii.gz"
        result_mask = find_all(file_type, nifti_dir)

        result_mask = result_mask if n_sample is None else result_mask[:n_sample]

        n1_mask = nib.load(result_mask[0]).get_fdata()

        n1_mask = torch.from_numpy(n1_mask).to(torch.int)
        n1_mask = n1_mask.unsqueeze(0)

        return n1_img, n1_mask
    else:  # no need to extract mask, useful for t1 linear registered image extraction
        return n1_img


# load nifti image to numpy


def get_numpy_from_nifti(nifti_dir, file_type="T1_brain_to_MNI.nii.gz", n_sample=None, save_numpy=False, save_numpy_dir=None):

    result = find_all(file_type, nifti_dir)

    result = result if n_sample is None else result[:n_sample]

    n1_img = nib.load(result[0])
    b, d, w, h = len(result), *n1_img.shape
    np_array = np.zeros((b, d, w, h),)
    for i, nifi_file in enumerate(result):
        # print(nib.load(nifi_file).get_fdata())
        np_array[i] = nib.load(nifi_file).get_fdata()
    if save_numpy:
        if save_numpy_dir:
            np.save(save_numpy_dir, np_array)

    return np_array


def save_binary_file(file_path, file_name):
    with open(file_path, 'wb') as f:
        pkl.dump(file_name, f)
############################## FPN Network ####################################


def create_train_val_test_set(image_root_path, train_test_val_dir, train_frac=0.8, val_frac=0.1):

    t1_raw_img_path = image_root_path
    img_paths = [os.path.join(t1_raw_img_path, idx)
                 for idx in os.listdir(t1_raw_img_path)]

    train_frac = train_frac
    val_frac = val_frac

    tot_sample = len(img_paths)
    train_spls = img_paths[0:int(tot_sample*train_frac)]
    print(f"Train samples: {len(train_spls)}")

    val_spls = img_paths[int(
        tot_sample*train_frac): int(tot_sample*train_frac) + int(tot_sample*val_frac)]

    print(f"Val samples: {len(val_spls)}")

    test_spls = img_paths[int(tot_sample*train_frac) +
                          int(tot_sample*val_frac):]

    print(f"Val samples: {len(test_spls)}")

    folder_names = ['train', 'val', 'test']
    sample_ls = [train_spls, val_spls, test_spls]

    for folder_name, samples in zip(folder_names, sample_ls):

        dest_dir = os.path.join(train_test_val_dir, f"{folder_name}")
        os.makedirs(dest_dir, exist_ok=True)
        for src_path in samples:
            # print(src_path)
            folder_name = os.path.basename(src_path)
            dest_path = os.path.join(dest_dir, folder_name)
            shutil.move(src_path, dest_path)


'''
root_path = os.path.dirname(os.path.dirname('<path/to>/project_root/'))
image_root_path = os.path.join(root_path,"data","raw_data","T1_orig")

train_test_val_dir = image_root_path
create_train_val_test_set(image_root_path = image_root_path, train_test_val_dir = train_test_val_dir)
'''

############################### DOWNSTREAM TASKS########################

# ------------------ Voxel Classification----------------


def save_fast_firstseg_files(image_id, hub_dir, save_dir, suffix_dest_path='2_0_T1_first_all_fast_firstseg.nii.gz',
                             serch_pattern="T1/T1_first/T1_first_all_fast_firstseg.nii.gz"):
    # image_id = "<subject_id>"
    # hub_dir = '<path/to>/ukb_t1_source'
    # save_dir = "<path/to>/tmp/"

    src_path = os.path.join(hub_dir, f"{image_id}_2_0.zip")

    dest_dir = os.path.join(save_dir, image_id)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"{image_id}_{suffix_dest_path}")

    with zipfile.ZipFile(src_path) as z:
        with z.open(serch_pattern) as zf, open(dest_path, 'wb') as f:
            shutil.copyfileobj(zf, f)
    print(f"Done for {dest_path}")


def extract_native_brain_masks_files(save_dir, serch_pattern='T1/T1_brain_mask.nii.gz',
                                     save_file_suffix="brain_mask.nii.gz",
                                     hub_dir='<path/to>/ukb_t1_source'):

    # root_path = config.root_path()

    os.makedirs(save_dir, exist_ok=True)

    all_fls = [f for f in os.listdir(hub_dir) if '20252_2_0' in f]
    print(f"Total found: {hub_dir}=> {len(all_fls)}")

    for file_id_zip in all_fls:

        type_dir_path = os.path.join(hub_dir, file_id_zip)
        file_id = file_id_zip.split(".")[0]

        dest_file_name = f"{file_id}_{save_file_suffix}"
        dest_file_path = os.path.join(save_dir, dest_file_name)

        try:
            with zipfile.ZipFile(type_dir_path) as z:
                with z.open(serch_pattern) as zf, open(dest_file_path, 'wb') as f:
                    shutil.copyfileobj(zf, f)
        except:
            print(f"No T1_brain_mask.nii.gz for {type_dir_path}")

        print(f"Done for {file_id_zip} => {dest_file_path}")


def extract_save_first_seg_files(hub_dir='<path/to>/ukb_t1_source'):

    root_path = config.root_path()

    label_dir = os.path.join(
        root_path, "data", "downstream_tasks", "voxel_classification")
    os.makedirs(label_dir, exist_ok=True)

    train_dir = os.path.join(root_path, "data", "raw_data", "T1_orig", "train")
    val_dir = os.path.join(root_path, "data", "raw_data", "T1_orig", "val")
    test_dir = os.path.join(root_path, "data", "raw_data", "T1_orig", "test")

    for dir_type, dir_path in zip(['train', 'val', 'test'], [train_dir, val_dir, test_dir]):
        type_dir_path = os.path.join(label_dir, dir_type)
        os.makedirs(type_dir_path, exist_ok=True)
        print(f"working on {dir_type}")
        files = os.listdir(dir_path)
        for image_id in files:
            save_fast_firstseg_files(
                image_id=image_id, hub_dir=hub_dir, save_dir=type_dir_path)


label_map_voxel_classification = {0.: 0, 10.: 1, 11.: 2, 12.: 3, 13.: 4, 16.: 5, 17.: 6, 18.: 7, 26.: 8, 49.: 9, 50.: 10, 51.: 11,
                                  52.: 12,
                                  53.: 13, 54.: 14, 58.: 15}


def get_image(image_id, image_dir, file_type="T1_brain_to_MNI.nii.gz",
              n_sample=1, save_numpy=False, save_numpy_dir=None):

    # get image
    image_path = os.path.join(image_dir, image_id)

    """
    img,img_mask = preprocessing.get_torch_from_single_nifti(nifti_dir = image_path,  
                                             file_type= file_type,
                                             n_sample=n_sample,
                                             save_numpy= save_numpy,
                                             save_numpy_dir=save_numpy_dir )
    """

    # print(f"image_path : {image_path}")
    # print(f"file type: {file_type}")
    result = find_all(file_type, image_path)

    result = result if n_sample is None else result[:n_sample]
    # print(result)
    n1_img = nib.load(result[0]).get_fdata()
    original_shape = n1_img.shape
    # print(f"imag.shape:{n1_img.shape}")
    # normalize it
    pre_pad, img = image_utils.get_suitable_image([n1_img])
    pre_pad = pre_pad[0]
    img = img[0]
    # print(f" after imag.shape:{img.shape}")
    # 2. Normalize it
    # img = img/img.max()
    img = torch.from_numpy(img)

    max_intensity = torch.max(img)
    min_intensity = torch.min(img)
    mri_data_rescaled = (img - min_intensity) / (max_intensity - min_intensity)
    img = mri_data_rescaled.unsqueeze(0).unsqueeze(0)
    img = img.to(torch.float)

    return img, original_shape


def get_mask_dict(image_id, label_dir):
    label_path = os.path.join(
        label_dir, image_id, f"{image_id}_2_0_T1_first_all_fast_firstseg.nii.gz")
    label_nii = nib.load(label_path).get_fdata()

    label_dict = {}
    labels = np.unique(label_nii)
    label_map = {}
    for lable in labels:

        # print(f"lable: {lable}")
        x, y, z = np.where(label_nii == lable)
        np_mat = np.asarray([x, y, z]).T
        label_dict[lable] = np_mat

    return label_nii, label_dict


def get_points_labels(label_dict, n=20, label_map_voxel_classification=label_map_voxel_classification):

    # n = 10
    selected_points_ls = []
    labels = []

    for label, points in label_dict.items():
        indexes = list(range(len(points)))
        random.shuffle(indexes)
        selected_point_indices = indexes[:n]
        selected_points = points[selected_point_indices]
        point_labels = [label_map_voxel_classification[label]]*n

        for point, lab in zip(selected_points, point_labels):
            selected_points_ls.append(point)
            labels.append(lab)

        points_np = np.vstack(selected_points_ls)
        lables_np = np.asarray(labels)

    return points_np, lables_np


def prepare_a_sample_voxel_classification(image_id, image_dir, label_dir, file_type="T1_brain_to_MNI.nii.gz",
                                          no_points_per_region=20, label_map_voxel_classification=label_map_voxel_classification):

    # get image
    img, original_shape = get_image(image_id=image_id, image_dir=image_dir, file_type=file_type,
                                    n_sample=1, save_numpy=False, save_numpy_dir=None)
    img = img.squeeze(dim=0)
    # get points to be predicted and associated lable
    label_nii, label_dict = get_mask_dict(image_id, label_dir)
    points_np, lables_np = get_points_labels(
        label_dict=label_dict, n=no_points_per_region, label_map_voxel_classification=label_map_voxel_classification)

    return img, points_np, lables_np, torch.tensor(original_shape)


class patch_dataset_from_file_voxel_classification(Dataset):

    def __init__(self, imgs_dir, label_dir, no_samples=-1,
                 file_type="T1_brain.nii.gz", no_points_per_region=20
                 ):
        super(patch_dataset_from_file_voxel_classification, self).__init__()
        self.imgs_dir = imgs_dir
        self.label_dir = label_dir

        self.imgs_paths = os.listdir(imgs_dir)
        random.shuffle(self.imgs_paths)

        if no_samples != -1:
            self.imgs_paths = self.imgs_paths[:no_samples]
        self.label_dir = label_dir

        self.file_type = file_type
        self.no_points_per_region = no_points_per_region

    def __len__(self):
        return len(self.imgs_paths)

    def __getitem__(self, idx):

        img_path = self.imgs_paths[idx]

        img, points_np, lables_np, original_shapep = prepare_a_sample_voxel_classification(image_id=img_path, image_dir=self.imgs_dir,
                                                                                           label_dir=self.label_dir, file_type=self.file_type,
                                                                                           no_points_per_region=self.no_points_per_region)

        return (img, points_np, lables_np, original_shapep)


############### Age Prediction#############################

def create_dataset_age_prediction(gwas_folder_name=None, suffix_file=None, train_sample_cnt=2000, val_sample_cnt=5000):
    ''' It creates the csv file containing the absolute path of the image and lable. It basically merge two csv files.
        the two csv files came from Khush.
    '''
    imag_df = pd.read_csv(os.path.join(
        config.data_path(), "T1_128_gwas.csv"))  # 37376 samples

    age_df = pd.read_csv(os.path.join(config.data_path(),
                         "T1_age_sex.csv"))  # 35868 samples
    age_df['eid'] = age_df['mri_names'].apply(
        lambda x: x.split("_")[0]).astype(np.int64)

    merged = pd.merge(imag_df, age_df, how='inner', left_on='EID', right_on='eid').drop(
        'mri_names_y', axis=1).rename({"mri_names_x": "mri_names"})

    # split
    indexes = list(range(len(merged)))
    random.shuffle(indexes)
    train_index = indexes[:train_sample_cnt]
    val_index = indexes[train_sample_cnt:(train_sample_cnt+val_sample_cnt)]
    test_index = indexes[(train_sample_cnt+val_sample_cnt):]

    train_df = merged.iloc[train_index]
    val_df = merged.iloc[val_index]
    test_df = merged.iloc[test_index]
    # save
    if gwas_folder_name is None:
        df_save_dir = config.data_path()
    else:
        df_save_dir = os.path.join(
            config.data_path(), 'downstream_tasks', gwas_folder_name)
        os.makedirs(df_save_dir, exist_ok=True)
    print(f'Csv files will be saved in {df_save_dir}')
    print(f"# Train Samples: {len(train_df)}")
    print(f"# Validation Samples: {len(val_df)}")
    print(f"# Test Samples: {len(test_df)}")

    if suffix_file is None:
        train_csv_file_name = "age_prediction_train.csv"
        val_csv_file_name = "age_prediction_val.csv"
        test_csv_file_name = "age_prediction_test.csv"
    else:
        train_csv_file_name = f"{suffix_file}_age_prediction_train.csv"
        val_csv_file_name = f"{suffix_file}_age_prediction_val.csv"
        test_csv_file_name = f"{suffix_file}_age_prediction_test.csv"

    train_df.to_csv(os.path.join(
        df_save_dir, train_csv_file_name), index=False)
    val_df.to_csv(os.path.join(df_save_dir, val_csv_file_name), index=False)
    test_df.to_csv(os.path.join(df_save_dir, test_csv_file_name), index=False)


class dataset_age_prediction(Dataset):

    def __init__(self, csv_file,
                 ):

        super(dataset_age_prediction, self).__init__()
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # we need to send the brain mask to select voxels from the foreground (as 78~ of the voxels are from background)
        img_path = self.df['mri_names_x'][idx]
        age = self.df['21003'][idx]

        # print(f"in the dataset img shape is :{img.shape}")
        # patch_list : list of tensor ; positive_pairs_ls : list of tensor

        # original image
        n1_img = nib.load(img_path).get_fdata()
        original_shape = torch.tensor(tuple(n1_img.shape))

        pre_pad, img = image_utils.get_suitable_image([n1_img])
        pre_pad = pre_pad[0]
        img = img[0]

        n1_img = torch.from_numpy(img)

        n1_img = n1_img.unsqueeze(0)
        max_intensity = torch.max(n1_img)
        min_intensity = torch.min(n1_img)
        mri_data_rescaled = (n1_img - min_intensity) / \
            (max_intensity - min_intensity)

        n1_img = mri_data_rescaled.to(torch.float)

        # brain mask
        brain_mask_path = self.df['mask_path'][idx]
        brain_mask = nib.load(brain_mask_path).get_fdata()

        # there are some number less 1 but greter than zero (they are from background, so nonzero() and np.where(s==1))
        # disagree
        # it works for both brain mask(in native space) and brain segmentation mask(standard space)
        brain_mask = np.where(brain_mask >= 1, 1, 0)
        x, y, z = np.where(brain_mask == 1)
        unraveled_indices = torch.from_numpy(
            np.ravel_multi_index((x, y, z), brain_mask.shape))
        return (n1_img, age, original_shape, unraveled_indices, brain_mask)


def make_suitable_shape(shape):

    padding_amt = [32 - p % 32 if p % 32 != 0 else 0 for p in shape]

    padding_amt = [(p/2, p/2) if p % 2 == 0 else ((p+1)/2, p - (p+1)/2)
                   for p in padding_amt]
    padding_amt = [(int(p[0]), int(p[1])) for p in padding_amt]

    return padding_amt


class dataset_age_prediction_whole(Dataset):

    def __init__(self, csv_file,
                 ):

        super(dataset_age_prediction_whole, self).__init__()
        self.df = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # we need to send the brain mask to select voxels from the foreground (as 78~ of the voxels are from background)
        img_path = self.df['mri_names_x'][idx]
        age = self.df['21003'][idx]

        # print(f"in the dataset img shape is :{img.shape}")
        # patch_list : list of tensor ; positive_pairs_ls : list of tensor

        # original image
        n1_img = nib.load(img_path).get_fdata()
        original_shape = torch.tensor(tuple(n1_img.shape))
        padding_amt = make_suitable_shape(n1_img.shape)

        pre_pad, img = image_utils.get_suitable_image([n1_img])
        pre_pad = pre_pad
        img = img[0]

        n1_img = torch.from_numpy(img)
        max_intensity = torch.max(n1_img)
        min_intensity = torch.min(n1_img)
        mri_data_rescaled = (n1_img - min_intensity) / \
            (max_intensity - min_intensity)

        n1_img = mri_data_rescaled.unsqueeze(0)
        # n1_img = n1_img/n1_img.max()
        n1_img = n1_img.to(torch.float)

        # brain mask
        brain_mask_path = self.df['mask_path'][idx]
        brain_mask = nib.load(brain_mask_path).get_fdata()

        # there are some number less 1 but greter than zero (they are from background, so nonzero() and np.where(s==1))
        # disagree
        # it works for both brain mask(in native space) and brain segmentation mask(standard space)
        brain_mask = np.where(brain_mask >= 1, 1, 0)
        x, y, z = np.where(brain_mask == 1)
        unraveled_indices = torch.from_numpy(
            np.ravel_multi_index((x, y, z), brain_mask.shape))

        brain_mask = np.pad(brain_mask, padding_amt, mode='constant')
        return (n1_img, age, original_shape, padding_amt, brain_mask)


class dataset_age_prediction_regressor(Dataset):

    def __init__(self, x, y
                 ):

        super(dataset_age_prediction_regressor, self).__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        return (self.x[idx], self.y[idx])


if __name__ == "__main__":
    print("d")
