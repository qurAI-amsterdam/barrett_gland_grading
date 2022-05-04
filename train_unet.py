from distutils.dir_util import copy_tree
import shutil
import torch
import torch.nn as nn
import numpy as np
from nn_archs import UNet
from wholeslidedata.iterators import create_batch_iterator
from label_utils import to_dysplastic_vs_non_dysplastic
import os
import argparse
from utils import print_dataset_statistics
import yaml
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb
from utils import mean_metrics

"""
To-do: Train segmentation network for Non-Dysplastic vs Dysplastic
(*) define a train and validation split
(*) adjust sampling strategy (with the wholeslidedata package)
(*) normalize data
(*) add data augmentation
    - HookNet: spatial, color, noise and stain augmentation (Tellez, 2018: Whole-Slide Mitosis Detection)
    - RaeNet: gamma transform, random flipping, Gaussian blur, affine translation and colour distortion on the training data
(*) decrease learning rate 
    - RaeNet: decreasing the learning rate by a factor of 0.01 every 50 epochs
(?) check input type: float32 vs float64: is there any difference?
(?) add class weights
(?) add dice loss
(?) plot intermediate predictions in weights and biases
"""


def load_config(user_config):
    with open(user_config, 'r') as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)

    return data['unet']


def train(run_name, experiments_dir, wandb_key):
    # config path
    base_dir = '/home/mbotros/code/barrett_gland_grading/'
    user_config = os.path.join(base_dir, 'configs/unet_training_config.yml')

    # make experiment dir & copy source files (config and training script)
    exp_dir = os.path.join(experiments_dir, run_name)
    print('Experiment stored at: {}'.format(exp_dir))
    copy_tree(os.path.join(base_dir, 'configs'), os.path.join(exp_dir, 'src', 'configs'))
    copy_tree(os.path.join(base_dir, 'nn_archs'), os.path.join(exp_dir, 'src', 'nn_archs'))
    shutil.copy2(os.path.join(base_dir, 'train_unet.py'), os.path.join(exp_dir, 'src'))

    # load network config and store in experiment dir
    print('Loaded config: {}'.format(user_config))
    train_config = load_config(user_config)

    # create train and validation generators (no reset)
    training_batch_generator = create_batch_iterator(user_config=user_config,
                                                     mode='training',
                                                     cpus=train_config['cpus'])

    validation_batch_generator = create_batch_iterator(mode='validation',
                                                       user_config=user_config,
                                                       cpus=train_config['cpus'])

    print('\nTraining dataset ')
    print_dataset_statistics(training_batch_generator.dataset)
    print('\nValidation dataset ')
    print_dataset_statistics(validation_batch_generator.dataset)
    print('\n')

    # create model and put on device(s)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=train_config['n_channels'], n_classes=train_config['n_classes'])
    model = nn.DataParallel(model) if train_config['gpus'] > 1 else model
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # log with weights and biases
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb.init(project="Barrett's Gland Grading", dir=exp_dir)
    wandb.run.name = run_name

    min_val = float('inf')

    for n in range(train_config['epochs']):

        train_metrics = {}
        validation_metrics = {}

        for idx in tqdm(range(train_config['train_batches']), desc='Epoch {}'.format(n + 1)):
            x, y, info = next(training_batch_generator)

            # dysplastic vs non-dysplastic
            y = to_dysplastic_vs_non_dysplastic(y)

            # transform x and y
            x = torch.tensor(x.astype('float32'))
            x = torch.transpose(x, 1, 3).to(device)
            y = torch.tensor(y.astype('int64')).to(device)

            # forward and update
            optimizer.zero_grad()
            y_hat = model.forward(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            # compute and store metrics
            y = y.cpu().detach().numpy().flatten()
            y_hat = torch.argmax(y_hat, dim=1).cpu().detach().numpy().flatten()
            train_metrics[idx] = {'loss': loss.item(),
                                  'dice per class': f1_score(y, y_hat, average=None),
                                  'dice weighted': f1_score(y, y_hat, average='weighted')}

        # validate
        with torch.no_grad():
            for idx in tqdm(range(train_config['val_batches']), desc='Validating'):
                x, y, info = next(validation_batch_generator)
                # dysplastic vs non-dysplastic
                y = to_dysplastic_vs_non_dysplastic(y)

                # transform x and y
                x = torch.tensor(x.astype('float32'))
                x = torch.transpose(x, 1, 3).to(device)
                y = torch.tensor(y.astype('int64')).to(device)

                # forward and validate
                y_hat = model.forward(x)
                loss = criterion(y_hat, y)

                # compute dice
                y = y.cpu().detach().numpy().flatten()
                y_hat = torch.argmax(y_hat, dim=1).cpu().detach().numpy().flatten()
                validation_metrics[idx] = {'loss': loss.item(),
                                           'dice per class': f1_score(y, y_hat, average=None),
                                           'dice weighted': f1_score(y, y_hat, average='weighted')}

        # compute and print metrics
        training_means = mean_metrics(train_metrics)
        validation_means = mean_metrics(validation_metrics)
        print("Train loss: {:.3f}, val loss: {:.3f}".format(training_means['loss'], validation_means['loss']))
        print("Train dice: {}, val dice: {}".format(np.round(training_means['dice per class'], decimals=2),
                                                    np.round(validation_means['dice per class'], decimals=2)))
        wandb.log({'epoch': n + 1,
                   'train loss': training_means['loss'], 'train dice': training_means['dice weighted'],
                   'val loss': validation_means['loss'], 'val dice': validation_means['dice weighted']})

        # save best model
        if validation_means['loss'] < min_val:
            torch.save(model.state_dict(),
                       os.path.join(exp_dir, 'model_epoch_{}_loss_{:.3f}.pt').format(n, validation_means['loss']))
            min_val = validation_means['loss']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='test', help="the name of this experiment")
    parser.add_argument("--exp_dir", type=str, default='/home/mbotros/experiments/barrett_gland_grading',
                        help="experiment dir")
    parser.add_argument("--wandb_key", type=str, help="key for logging to weights and biases")
    args = parser.parse_args()
    train(args.run_name, args.exp_dir, args.wandb_key)
