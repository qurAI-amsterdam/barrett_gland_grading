import numpy as np


def to_dysplastic_vs_non_dysplastic(y_batch):
    """Simplifies the segmentation problem by setting: non-dysplastic (label = 1) vs dysplastic (label = 2)
    Parameters:
    y_batch: input batch labels (np.array)

    returns: y_batch (np.array)
    """
    # set lgd and hgd as dysplastic
    y_batch = np.where(y_batch > 1, 2, y_batch)
    return y_batch