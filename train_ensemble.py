import argparse
from train_segmentation import train
import os
from random import randrange
import yaml
from utils import load_config
from train_segmentation import load_segmentation_model
import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import shutil


def load_model(model_path, train_config, device=None):
    """ Load a model

    """
    model = load_segmentation_model(train_config, activation=None)
    model.load_state_dict(torch.load(model_path))
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()
    print('Loaded model from: {}\n'.format(model_path))

    return model


class Ensemble:
    """ An ensemble of models.

    """

    def __init__(self, ensemble_exp_dir, device, m=5):

        # load base config
        self.user_config = os.path.join(ensemble_exp_dir, 'base_config.yml')
        _, self.train_config = load_config(self.user_config)
        self.nets = sorted([os.path.join(ensemble_exp_dir, n, 'checkpoints', 'best_model.pt') for n in os.listdir(ensemble_exp_dir) if 'net' in n])[:m]
        self.device = device
        self.models = [load_model(n, self.train_config, self.device) for n in self.nets]

    def forward(self, x, y):
        """ Forwards a batch of images through the ensemble.

        Args:
            x:
                torch.tensor: (B, H, W)

        Returns:
            y_avg_prob: logits
                np.array (B, C, H, W)
        """

        # for averaging the M predictions
        y_pred_patches = []  # elements are (M, C, H, W)

        for i, model in enumerate(self.models):
            # forward, convert logits to probabilities
            y_pred_batch_m = model(x)
            y_pred_batch_m_soft = torch.nn.functional.softmax(y_pred_batch_m, dim=1).cpu().detach().numpy()
            y_pred_patches.append(y_pred_batch_m_soft[:len(y)])

        # stack patches: (M, B, C, H, W)
        y_pred_stack = np.stack(y_pred_patches, axis=0)

        # average probabilities: (B, C, H, W)
        y_avg_prob = np.mean(y_pred_stack, axis=0)

        return y_avg_prob


def train_ensemble(m, ensemble_run_name, exp_dir, wandb_key):
    """ Trains and ensemble of M segmentation models with

    Confidence Calibration and Uncertainty Estimation for Deep Medical Image Segmentation
    (https://arxiv.org/abs/1911.13273)

    (1) Random initialization of net parameters
    (2) random shuffling the (same) training data

    Args:
        m: the number of networks in the ensemble
        ensemble_run_name: name of the ensemble run
        exp_dir: location where to store experiment info
        wandb_key: wandb key for logging on W&B
    """

    # the base config
    base_dir = '/home/mbotros/code/barrett_gland_grading/'
    base_config = os.path.join(base_dir, 'configs/base_config.yml')

    # copy code and base config
    run_dir = os.path.join(exp_dir, ensemble_run_name)
    os.makedirs(run_dir, exist_ok=True)
    shutil.copy2(os.path.join(base_dir, 'configs/base_config.yml'), run_dir)
    shutil.copy2(os.path.join(base_dir, 'train_segmentation.py'), run_dir)
    shutil.copy2(os.path.join(base_dir, 'train_ensemble.py'), run_dir)

    for i in range(m):

        # take a random seed for data shuffling
        seed = randrange(100)
        print('Training net {} with data shuffle seed {}.'.format(i, seed))

        # make dir for this net
        ensemble_dir = os.path.join(exp_dir, ensemble_run_name, 'net_{}/'.format(i))
        os.makedirs(ensemble_dir, exist_ok=True)
        print('stored at {}'.format(ensemble_dir))

        # for each net adapt the seed in the config file
        with open(base_config, 'r') as yamlfile:
            net_config_path = os.path.join(ensemble_dir, 'user_config.yml')
            data = yaml.load(yamlfile, Loader=yaml.FullLoader)
            data['wholeslidedata']['default']['seed'] = seed
            with open(net_config_path, 'w') as outfile:
                yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

        wandb_run_name = ensemble_run_name + 'net_{}'.format(i)
        train(run_name=wandb_run_name, exp_dir=ensemble_dir, wandb_key=wandb_key)
        print('Finished training net {}!\n'.format(i))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_run_name", type=str, default='m2_test', help="the name of this experiment")
    parser.add_argument("--exp_dir", type=str,
                        default='/data/archief/AMC-data/Barrett/experiments/barrett_gland_grading/3_classes/',
                        help="experiment dir")
    parser.add_argument("--wandb_key", type=str, help="key for logging to weights and biases")
    parser.add_argument("--m", type=int, default='5', help="the number of nets in the ensemble")
    args = parser.parse_args()
    train_ensemble(args.m, args.ensemble_run_name, args.exp_dir, args.wandb_key)
