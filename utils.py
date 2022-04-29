from wholeslidedata.samplers.utils import plot_mask
from matplotlib import pyplot as plt
from pprint import pprint


# dataset utilities
def print_dataset_statistics(dataset):
    pprint(dataset.annotations_per_label_per_key)

    print('\nAnnotation level: {}'.format(dict(sorted(dataset.annotations_per_label.items()))))
    print('Pixel level: {}'.format(dict(sorted(dataset.pixels_per_label.items()))))

    # compute pixel ratio's in this dataset
    total_pixels = sum(dataset.pixels_per_label.values())
    percentage_dict = {k: (v * 100 / total_pixels) for k, v in sorted(dataset.pixels_per_label.items())}
    print('Pixel percentage per class: {:.2f}'.format(percentage_dict))


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


