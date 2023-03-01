import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import colors_1, colors_2
from wholeslidedata.samplers.utils import plot_mask
from metrics_lib import _validate_probabilities
from sklearn.metrics import log_loss


def calc_bins(y_true, y_pred, num_bins=10):
    """ Compute bins for ECE and reliability plot.

    Args:
        y_true: ground truth labels
            (N)
        y_pred: predicted probabilities
            (N, C)
        num_bins: the number of bins for the plot

    Returns:

    """
    # validate probabilities
    _validate_probabilities(y_pred)

    # Assign each prediction to a bin
    bins = np.linspace(0.1, 1, num_bins)    # (num_bins,)
    binned = np.digitize(y_pred, bins)      # (N, C) value: ind to which bin it belong

    # Convert labels to one hot
    num_classes = y_pred.shape[1]
    y_true = np.eye(num_classes)[y_true]

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for b in range(num_bins):
        bin_sizes[b] = (binned == b).sum()

        if bin_sizes[b] > 0:
            bin_accs[b] = (y_true[binned == b]).sum() / bin_sizes[b]
            bin_confs[b] = (y_pred[binned == b]).sum() / bin_sizes[b]

    return bins, bin_accs, bin_confs, bin_sizes


def nll_score(y_true, y_pred):
    """ Own implementation of NLL.

    Args:
        y_true: (N,)
        y_pred: (N, C)

    Returns:
        nll: (1, )

    """
    # validate probs
    _validate_probabilities(y_pred)

    num_samples = y_pred.shape[0]
    num_classes = y_pred.shape[-1]

    # to one hot
    y_true = np.eye(num_classes)[y_true]

    # compute nll
    nll = -np.sum(np.log(y_pred) * y_true) / num_samples
    return nll


def brier_score(y_true, y_pred):
    """ Computes the normalized Brier score.

    Args:
        y_true: (N,)
        y_pred: (N, C)

    Returns:
        norm_brier_score: (1, )

    """
    # validate probs
    _validate_probabilities(y_pred)
    num_classes = y_pred.shape[-1]

    # to one hot
    y_true = np.eye(num_classes)[y_true]

    # compute normalized brier score
    norm_brier_score = np.mean(np.square(y_pred - y_true))
    return norm_brier_score


def ece(y_true, y_pred):
    """ Computes the Expected Calibration Error.

    Args:
        y_true: (N, )
        y_pred: (N, C)

    Returns:
        ece: (1, )
    """
    ece = 0
    bins, bin_accs, bin_confs, bin_sizes = calc_bins(y_true, y_pred)

    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ece += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif

    return ece


def avg_entropy_sk_per_patch(y_pred):
    """ Compute the average entropy per patch in a batch

    Args:
        y_pred: (B, C, H, W)

    Returns:
        h: (B, C)
    """
    h = np.zeros((y_pred.shape[0], y_pred.shape[1]))

    for i, y_pred_patch in enumerate(y_pred):
        y_pred_patch = np.transpose(y_pred_patch, (1, 2, 0)).reshape(-1, y_pred.shape[1])
        h[i] = avg_entropy_sk(y_pred_patch)

    return h


def avg_entropy(y_pred, epsilon=1e-5):
    """ Computes the average of pixel-wise entropy values for all pixels.

    Args:
        y_pred: (N, C)
        epsilon: small number for computation

    Returns:
        avg_entropy_sk: (C, )
    """
    # validate probabilities
    _validate_probabilities(y_pred)
    num_classes = y_pred.shape[1]
    avg_entropy = np.zeros(num_classes)

    for c in range(num_classes):

        # for every pixel the prob of c
        p_c = y_pred[:, c]
        avg_entropy[c] = -(1 / len(p_c)) * np.sum(p_c * np.log(p_c + epsilon) + (1 - p_c) * np.log(1 - p_c + epsilon))

    return avg_entropy


def avg_entropy_sk(y_pred, epsilon=1e-5):
    """ Computes the average of pixel-wise entropy values over the predicted foreground.

    Args:
        y_pred: (N, C)
        epsilon: small number for computation

    Returns:
        avg_entropy_sk: (C, )
    """
    # validate probabilities
    _validate_probabilities(y_pred)

    y_pred_hard = np.argmax(y_pred, axis=1)
    num_classes = y_pred.shape[1]
    avg_entropy = np.zeros(num_classes)
    max_entropy = np.log(2) * num_classes

    for c in range(num_classes):

        # take the probabilities of predicted foreground class c
        p_c = y_pred[y_pred_hard == c]

        if len(p_c) == 0:
            avg_entropy[c] = max_entropy
        else:
            avg_entropy[c] = -(1 / len(p_c)) * np.sum(
                p_c * np.log(p_c + epsilon) + (1 - p_c) * np.log(1 - p_c + epsilon))

    return avg_entropy / max_entropy


def plot_reliability_diagram(y_true, y_pred, ax):
    """ Plots the reliability diagram for one patch.

    Args:
        y_true: [N, ]
        y_pred: [N, C]
        ax:

    Returns:
    """
    bins, bin_accs, bin_confs, bin_sizes = calc_bins(y_true, y_pred)

    # x/y limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True)
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(bins, bins,  width=-0.1, alpha=0.4, edgecolor='black', color='r', hatch='\\', align='edge')

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=-0.1, alpha=1, edgecolor='black', color='b', align='edge')
    plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # Legend
    outputs_patch = mpatches.Patch(color='b', label='Outputs')
    gaps_patch = mpatches.Patch(color='r', alpha=0.4, hatch='\\', label='Gaps')

    plt.legend(handles=[outputs_patch, gaps_patch], title='ECE: {:.2f}\nNLL: {:.2f}\nBrier: {:.2f}'.format(
        ece(y_true, y_pred) * 100,
        brier_score(y_true, y_pred),
        log_loss(y_true, y_pred, labels=[0, 1, 2, 3])))
    plt.show()
#     plt.savefig('calibrated_network.png', bbox_inches='tight')


def plot_class_probabilities_sample(y_true, y_pred):
    """ Plots the class probabilities vs No. of Pixels & a reliability diagram.

    Args:
        y_true: [N,]
        y_pred: [N, C]
    """

    # validate first
    _validate_probabilities(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(18, 4), gridspec_kw={'height_ratios': [2]})

    # plot class prob vs No of pixels
    y_pred_hard = np.argmax(y_pred, axis=1)
    axes[0].hist(y_pred[y_pred_hard == 1][:, 1], bins=50, label='NDBE', alpha=0.7, color='green')
    axes[0].hist(y_pred[y_pred_hard == 2][:, 2], bins=50, label='LGD', alpha=0.7, color='orange')
    axes[0].hist(y_pred[y_pred_hard == 3][:, 3], bins=50, label='HGD', alpha=0.7, color='red')
    axes[0].set_xlabel('Class Probability', fontsize=18)
    axes[0].set_ylabel('No. of pixels', fontsize=18)
    axes[0].legend(fontsize=15)
    axes[0].set_title('Average Entropy: {}'.format(
        tuple(np.round(avg_entropy_sk(y_pred), decimals=2)),
        fontsize=15))
    axes[0].tick_params(axis='both', labelsize=18, pad=5)

    # plot reliability diagram
    plot_reliability_diagram(y_true, y_pred, axes[1])


def plot_pred_sample(x, y, y_hat, save_path=None):
    """ Plots the center of a batch with prediction.
    The plot is stored in the experiment dir to keep track of performance.

    Args:
        x: [H, W, CHANNELS]
        y: [H, W]
        y_hat: [CLASSES, H, W]
        save_path: path where plot of predictions is stored.

    Returns:
        none: saves the figure at the save patch
    """
    patches = 1
    n_classes = y_hat.shape[0]

    # get the prediction
    y_hat_hard = np.argmax(y_hat, axis=0)

    # probability of dysplasia
    y_hat_dys = y_hat[1] + y_hat[2]
    y_hat_dys = np.ma.masked_where(y_hat_hard == 0, y_hat_dys)

    # show just the image
    fig, axes = plt.subplots(patches, 3, figsize=(18, 10), squeeze=False)

    if n_classes == 4:
        green_patch = mpatches.Patch(color='green', label='NDBE', alpha=0.5)
        orange_patch = mpatches.Patch(color='orange', label='LGD', alpha=0.5)
        red_patch = mpatches.Patch(color='red', label='HGD', alpha=0.5)
        plt.legend(handles=[green_patch, orange_patch, red_patch], bbox_to_anchor=(1.03, 1.0), loc='upper left',
                   borderaxespad=0., prop={'size': 10})
        colors = colors_1
    else:
        green_patch = mpatches.Patch(color='green', label='NDBE', alpha=0.5)
        red_patch = mpatches.Patch(color='red', label='DYS', alpha=0.5)
        plt.legend(handles=[green_patch, red_patch], bbox_to_anchor=(1.03, 1.0), loc='upper left', borderaxespad=0.,
                   prop={'size': 10})
        colors = colors_2

    # plot images
    plt.subplot(131)
    plt.imshow(x)
    plt.subplot(132)
    plt.imshow(x)
    plt.subplot(133)
    plt.imshow(x)
    # plt.subplot(144)
    # plt.imshow(x)

    # plot masks
    plot_mask(y, axes=axes[0][1], alpha=0.3, color_values=colors)
    plot_mask(y_hat_hard, axes=axes[0][2], alpha=0.3, color_values=colors)
    # axes[0][3].imshow(y_hat_dys, cmap='RdBu_r', alpha=0.3)

    # set titles
    axes[0][1].set_title('Ground Truth', fontsize=18)
    axes[0][2].set_title('Prediction', fontsize=18)
    axes[0][0].set_title('Tissue', fontsize=18)
    # axes[0][3].set_title('Prob Dysplasia', fontsize = '18')

    for ax in axes.flatten():
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()