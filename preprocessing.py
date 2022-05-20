import numpy as np
import albumentations as albu
import torch


def to_dysplastic_vs_non_dysplastic(y, **kwargs):
    """Simplifies the segmentation problem by setting: non-dysplastic (label = 1) vs dysplastic (label = 2).

    Args:
    y_batch: input batch labels (np.array)

    Returns:
        y_batch (np.array)
    """
    return np.where(y > 1, 2, y)


def to_tensor_image(x, **kwargs):
    return torch.tensor(x.astype('float32'))


def to_tensor_mask(x, **kwargs):
    return torch.tensor(x.astype('int64'))


def transpose(x, **kwargs):
    # [B, H, W, 3] => [B, 3, H, W]
    return x.transpose(0, 3, 1, 2)


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transforms.

    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
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
