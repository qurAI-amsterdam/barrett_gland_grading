from wholeslidedata.samplers.utils import plot_mask
from matplotlib import pyplot as plt
from pprint import pprint
import numpy as np
import matplotlib.patches as mpatches
import yaml
import pandas as pd
import seaborn as sns

# define some colors
colors_1 = ["white", "green", "orange", "red", 'yellow', 'yellow', 'purple', 'pink', 'grey', "blue"]
colors_2 = ["white", "green", "red", "yellow", 'brown', 'yellow', 'purple', 'pink', 'grey', 'green']


def load_config(user_config):
    with open(user_config, 'r') as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)

    return data['wholeslidedata'], data['net']


# dataset utilities
def print_dataset_statistics(dataset, show_all_files=False):

    if show_all_files:
        pprint(dataset.annotations_per_label_per_key)

    annotation_level_dict = dict(sorted(dataset.annotations_per_label.items()))
    pixel_level_dict = dict(sorted(dataset.pixels_per_label.items()))
    print('Annotation level: {}'.format(annotation_level_dict))
    print('Pixel level: {}'.format(pixel_level_dict))

    # compute pixel ratio's in this dataset
    total_pixels = sum(dataset.pixels_per_label.values())
    percentage_dict = {k: round((v * 100 / total_pixels), 2) for k, v in dataset.pixels_per_label.items()}
    percentage_dict = dict(sorted(percentage_dict.items()))
    print('Pixel percentage per class: {}'.format(percentage_dict))

    return {'annotation level': annotation_level_dict,
            'pixel level': pixel_level_dict,
            'percentage': percentage_dict}


# plotting utilities
def init_plot(batches, batch_size, size=(20, 5)):
    fig, axes = plt.subplots(batches, batch_size, figsize=size, squeeze=False)
    return fig, axes


def show_plot(fig, r, fontsize=16):
    fig.suptitle(f'Batches (repeat={r})', fontsize=fontsize)
    plt.tight_layout()
    plt.show()


def plot_batch(axes, idx, x_batch, y_batch, alpha=0.4, colors=None):
    for batch_index in range(len(x_batch)):
        axes[idx][batch_index].imshow(x_batch[batch_index])
        plot_mask(y_batch[batch_index], axes=axes[idx][batch_index], alpha=alpha, color_values=colors)


def mean_metrics(metrics_dict):
    mean = {}
    for m in metrics_dict[0].keys():
        s = 0
        for k, v in metrics_dict.items():
            s += v[m]
        mean[m] = s / len(metrics_dict)
    return mean


def crop_center(img, cropx, cropy):
    _, x, y, _ = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[:, starty:starty + cropy, startx:startx + cropx, :]


def plot_pred_batch(x, y, y_hat, save_path=None, patches=3, h_pad=0.5, w_pad=-28):
    """ Plots the center of a batch with prediction.
    The plot is stored in the experiment dir to keep track of performance.

    Args:
        x: [B, H, W, CHANNELS]
        y: [B, H, W]
        y_hat: [B, CLASSES, H, W]
        save_path: path where plot of predictions is stored.
        patches: how many patches to include in the plot
        h_pad: height padding
        w_pad: width padding
        plot: whether to plot or save

    Returns:
        none: saves the figure at the save patch
    """
    patches = min(len(x), patches)

    # get the prediction
    y_hat = np.argmax(y_hat, axis=1)

    # center crop the image
    _, h, w = y_hat.shape
    x = crop_center(x, h, w)

    green_patch = mpatches.Patch(color='green', label='NDBE', alpha=0.5)
    orange_patch = mpatches.Patch(color='orange', label='LGD', alpha=0.5)
    red_patch = mpatches.Patch(color='red', label='HGD', alpha=0.5)

    # show just the image
    fig, axes = plt.subplots(3, patches, figsize=(20, 14), squeeze=False)
    plt.legend(handles=[green_patch, orange_patch, red_patch], bbox_to_anchor=(1.03, 1.0), loc='upper left', borderaxespad=0.,  prop={'size': 10})
    plot_batch(axes, 0, x[:patches], np.zeros_like(y)[:patches], alpha=0.3, colors=colors_1)
    plot_batch(axes, 1, x[:patches], y[:patches], alpha=0.3, colors=colors_1)
    plot_batch(axes, 2, x[:patches], y_hat[:patches], alpha=0.3, colors=colors_1)
    for ax in axes.flatten():
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
    plt.tight_layout(h_pad=h_pad, w_pad=w_pad)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(cf_matrix, save_path=None):
    """ Plots the confusion matrix

    Args:
        cf_matrix: the confusion matrix to plot
        save_path: location where to store the plot
        plot: whether to plot it of save it

    Returns:
        none: saves the figure at the save path
    """

    df_cm = pd.DataFrame(cf_matrix, index=[i for i in ['BG', 'NDBE', 'LGD', 'HGD']],
                         columns=[i for i in ['BG', 'NDBE', 'LGD', 'HGD']])

    plt.figure(figsize=(15, 10))
    with sns.plotting_context(font_scale=2):
        sns.heatmap(df_cm, annot=True, cmap="Blues", square=True, fmt='.2f')
    plt.gca().set_yticklabels(labels=['BG', 'NDBE', 'LGD', 'HGD'], va='center')
    plt.gca().set_ylabel('True', labelpad=30)
    plt.gca().set_xlabel('Predicted', labelpad=30)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()