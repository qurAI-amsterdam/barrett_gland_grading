import os
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
from wholeslidedata.iterators import create_batch_iterator
import segmentation_models_pytorch as smp
import argparse
from preprocessing import get_preprocessing
from train_ensemble import Ensemble
from confidence_calibration import avg_entropy_sk_per_patch

# (1) image patches (top N)
# (2) segmentation embedding
# (3) h_sk
# (4) image net embedding
# (5) ssl embedding
# overlapping sliding window


def extract_tiles(model, batch_iterator, preprocessing, device):
    """ Extracts tiles.

    Args:
            model:
            batch_iterator:
            preprocessing:
            device:

    Returns:
            info_tiles:
    """
    info_tiles = []

    with torch.no_grad():
        for idx, (x_np, y_np, info) in enumerate(tqdm(batch_iterator)):

            # preprocessing
            sample = preprocessing(image=x_np, mask=y_np)
            x, y = sample['image'].to(device), sample['mask'].to(device)

            # forward: out is probs (B, C, H, W)
            y_hat = model.forward(x, y)

            # naive max grade prediction
            y_pred_max_grade = np.max(np.argmax(y_hat, axis=1), axis=(1, 2))  # (B, 1)
            y_prob_max_grade = np.mean(y_hat, axis=(2, 3))  # (B, C)

            # entropy (uncertainty score)
            avg_entropy = avg_entropy_sk_per_patch(y_hat)

            for i in range(len(y_hat)):
                sample_reference = info['sample_references'][i]
                file_key = sample_reference['reference'].file_key
                point = sample_reference['point']

                # print('Idx: {}, point:  {}, shape: {}: '.format(idx, point, x_np[i].shape))
                info_tiles.append({'file': file_key,
                                   'point': point,
                                   'naive pred': y_pred_max_grade[i],
                                   'naive prob': y_prob_max_grade[i],
                                   'entropy pred': avg_entropy[i][y_pred_max_grade[i]],
                                   'entropy_sk': np.round(avg_entropy[i], decimals=3)})

    return info_tiles


def extract_entropy_scores(extract_config):
    """
    """

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # 50% overlapping sliding window over the biopsy outlines
    print('Loading config: {}'.format(extract_config))
    extract_iterator = create_batch_iterator(mode='training',
                                             user_config=extract_config,
                                             cpus=1,
                                             number_of_batches=-1,
                                             return_info=True)

    # load Ensemble of 5 UNet++
    exp_dir = '/data/archief/AMC-data/Barrett/experiments/barrett_gland_grading/3_classes/Ensemble_m5_UNet++_CE_IN/'
    preprocessing = get_preprocessing(smp.encoders.get_preprocessing_fn('efficientnet-b4', 'imagenet'))
    ensemble_m5_CE_IN = Ensemble(exp_dir, device=device, m=5)

    # load RBE case level diagnosis
    rbe_slide_df = pd.read_csv('/data/archief/AMC-data/Barrett/labels/rbe_slide_level.csv')
    rbe_slide_df['grade normal'] = rbe_slide_df['grade'].map({'NDBE': 1, 'LGD': 2, 'HGD': 3})

    # extract h_sk for tiles
    info_tiles = extract_tiles(ensemble_m5_CE_IN, extract_iterator, preprocessing, device)
    df = pd.DataFrame(info_tiles)
    slides = np.unique(df['file'])
    slides = [slide for slide in slides if slide in list(rbe_slide_df['slide'])]

    # make a dataset
    x = np.zeros((len(slides), 25, 4))
    y = np.zeros((len(slides), 1))

    for idx, slide in enumerate(slides):

        # look up y
        y[idx] = int(rbe_slide_df[rbe_slide_df['slide'] == slide]['grade normal']) - 1

        # construct features: top 25 low entropy suspicous tiles
        slide_df = df[df['file'] == slide]
        suspicious_tiles = slide_df[slide_df['naive pred'] > 1].sort_values(by=['entropy pred'])[:25]

        for tile_idx, tile in suspicious_tiles.reset_index().iterrows():
            x[idx, tile_idx, :] = tile['entropy_sk']

    # store features
    classification_exp_dir = '/data/archief/AMC-data/Barrett/experiments/barrett_slide_classification/top_25_entropy'
    x_save_path = os.path.join(classification_exp_dir, 'x_train_overlap')
    y_save_path = os.path.join(classification_exp_dir, 'y_train_overlap')

    np.save(file=x_save_path, arr=x)
    np.save(file=y_save_path, arr=y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract_config", type=str,
                        default='/home/mbotros/code/barrett_gland_grading/configs/tile_extraction_config.yml',
                        help="experiment dir classification")
    args = parser.parse_args()
    extract_entropy_scores(args.extract_config)

