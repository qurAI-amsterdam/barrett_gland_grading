import numpy as np


def to_dysplastic_vs_non_dysplastic(y_batch):
    """Simplifies the segmentation problem by setting: dysplastic vs non dysplastic
    Parameters:
    y_batch: input batch labels (np.array)

    stroma (label = 1) => bg (label = 0)
    ndbe (label = 2) => non dysplastic (label = 1)
    lgd and hgd (label = 3, 4) => dysplastic (label = 2)

    returns: y_batch (np.array)
    """
    # set stroma as bg
    y_batch = np.where(y_batch == 1, 0, y_batch)

    # set ndbe as non-dysplastic
    y_batch = np.where(y_batch == 2, 1, y_batch)

    # set lgd and hgd as dysplastic
    y_batch = np.where(y_batch > 2, 2, y_batch)
    return y_batch