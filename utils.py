from wholeslidedata.samplers.utils import plot_mask
from matplotlib import pyplot as plt
from pprint import pprint


# dataset utilities
def print_dataset_statistics(dataset, show_all_files=False):

    if show_all_files:
        pprint(dataset.annotations_per_label_per_key)

    print('Annotation level: {}'.format(dict(sorted(dataset.annotations_per_label.items()))))
    print('Pixel level: {}'.format(dict(sorted(dataset.pixels_per_label.items()))))

    # compute pixel ratio's in this dataset
    total_pixels = sum(dataset.pixels_per_label.values())
    percentage_dict = {k: round((v * 100 / total_pixels), 2) for k, v in dataset.pixels_per_label.items()}
    print('Pixel percentage per class: {}'.format(dict(sorted(percentage_dict.items()))))


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
