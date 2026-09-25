"""Padding helpers shared by the dataset and the extraction pass.

The backbone downsamples by 32, so volumes are zero-padded up to the next
multiple of 32 in every spatial dimension. The amount padded before each axis is
returned as well, so coordinates can be mapped back to the original volume.
"""
import numpy as np


def make_suitable_shape(shape):
    """Per-axis (before, after) padding that rounds each extent up to a
    multiple of 32, split as evenly as possible."""
    padding_amt = [32 - p % 32 if p % 32 != 0 else 0 for p in shape]

    padding_amt = [(p / 2, p / 2) if p % 2 == 0 else ((p + 1) / 2, p - (p + 1) / 2)
                   for p in padding_amt]
    padding_amt = [(int(p[0]), int(p[1])) for p in padding_amt]

    return padding_amt


def get_suitable_image(img_list):
    """Zero-pad one image or a list of images to a backbone-compatible shape.

    Returns (pre_pad, images), where pre_pad holds the padding added before each
    axis. Both are lists, even when a single image is passed.
    """
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

    padding_amt = make_suitable_shape(img_list)
    pre = [p[0] for p in padding_amt]
    img = np.pad(img_list, padding_amt)

    return [pre], [img]
