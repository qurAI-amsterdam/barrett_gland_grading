import numpy as np
import albumentations as albu
import torch
from scipy import ndimage
from tiatoolbox.utils.misc import get_luminosity_tissue_mask


def to_dysplastic_vs_non_dysplastic(y, **kwargs):
    """Simplifies the segmentation problem by setting: non-dysplastic (label = 1) vs dysplastic (label = 2).

    Args:
        y: input batch labels (np.array).

    Returns:
        y (np.array): mask is now NDBE vs DYS.
    """
    return np.where(y > 1, 2, y)


def to_tensor_image(x, **kwargs):
    return torch.tensor(x.astype('float32'))


def to_tensor_mask(x, **kwargs):
    return torch.tensor(x.astype('int64'))


def transpose(x, **kwargs):
    # [B, H, W, 3] => [B, 3, H, W]
    return x.transpose(0, 3, 1, 2)


def filter_holes(tissue_w_holes, size_thresh):
    """Filters holes from a tissue mask.

    Args:
        tissue_w_holes: tissue mask with holes
        size_thresh: threshold for size of the removable holes

    Returns:
        tissue mask without holes

    (from: https://github.com/BPdeRooij/barrett_patch_extractor/)
    """
    # filter small objects from mask
    label_objects, _ = ndimage.label(tissue_w_holes)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes > size_thresh
    mask_sizes[0] = 0
    tissue_w_holes = mask_sizes[label_objects]

    # find holes using inverse and filter out large holes
    holes = np.invert(tissue_w_holes)
    label_objects, _ = ndimage.label(holes)
    sizes = np.bincount(label_objects.ravel())
    mask_sizes = sizes < size_thresh
    mask_sizes[0] = 0
    holes = mask_sizes[label_objects]

    return np.logical_or(tissue_w_holes, holes)


def tissue_mask_batch(x, y, lum_thresh=0.85, size_thresh=5000):
    """Luminosity based tissue masker.

    Args:
        x: batch of images
            shape: (B, H, W, C)
        y: batch of annotations that has to be tissue masked
            shape: (B, H, W)
        lum_thresh: threshold for luminosity tissue
        size_thresh: threshold for filtering hole sizes

    Returns:
        y_masked: batch of annotations that are tissue masked
            shape: (B, H, W)

    """
    # result array
    y_masked = np.zeros_like(y)

    for i in range(len(x)):

        # get a tissue mask & filter holes
        image, mask = x[i], y[i]
        tissue_mask = get_luminosity_tissue_mask(image, threshold=lum_thresh)
        tissue_mask = filter_holes(tissue_mask, size_thresh=size_thresh)

        # apply the tissue mask
        y_masked[i] = np.where(np.logical_and(tissue_mask, mask), mask, 0)

    return y_masked


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transforms.

    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network).
    Return:
        transform: albumentations.Compose

    """
    _transform = [albu.Lambda(image=preprocessing_fn)] if preprocessing_fn else []

    _transform.extend([
        albu.Lambda(image=transpose),
        albu.Lambda(mask=to_dysplastic_vs_non_dysplastic),
        albu.Lambda(image=to_tensor_image, mask=to_tensor_mask),
    ])
    return albu.Compose(_transform)
