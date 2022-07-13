import numpy as np
import tqdm
import os
import torch
import segmentation_models_pytorch as smp
from utils import load_config
import argparse
from train_segmentation import load_segmentation_model
from preprocessing import get_preprocessing


def load_trained_segmentation_model(exp_dir, model_path):
    """ Loads the trained model.

    Args:
        exp_dir: directory that hold all the information from an experiments (src, checkpoints)
        model_path: path to the trained model

    """
    user_config = os.path.join(exp_dir, 'src/configs/base_config.yml')
    _, train_config = load_config(user_config)

    # LOAD MODEL
    model = load_segmentation_model(train_config, activation=None)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print('Loaded model from {}'.format(model_path))

    # LOAD PREPROCESSING
    if train_config['encoder_weights']:
        preprocessing = get_preprocessing(smp.encoders.get_preprocessing_fn(
            train_config['encoder_name'], train_config['encoder_weights']))
    else:
        preprocessing = get_preprocessing()
    print('During training we used {} as encoder with weights from {}.'.format(train_config['encoder_name'],
                                                                               train_config['encoder_weights']))
    return model, preprocessing


def train(run_name, experiments_dir, wandb_key):
    """ Train something attention-based with segmentations outputs
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='test', help="the name of this experiment")
    parser.add_argument("--exp_dir_segmentation", type=str,
                        default='/data/archief/AMC-data/Barrett/experiments/barrett_gland_grading/3_classes',
                        help="experiment dir segmentation")
    parser.add_argument("--exp_dir_classification", type=str,
                        default='/data/archief/AMC-data/Barrett/experiments/barrett_gland_grading/3_classes',
                        help="experiment dir classification")
    parser.add_argument("--wandb_key", type=str, help="key for logging to weights and biases")
    args = parser.parse_args()
    train(args.run_name, args.exp_dir, args.wandb_key)
