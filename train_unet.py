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

"""
To-do:
(1) logging with weights and biases
(2) add validation metrics (F1 weighted, per label)
(3) adjust sampling strategy with the wholeslidedata package
(*) add class weights
(*) add dice loss
"""


def load_config(user_config):
    with open(user_config, 'r') as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)

    return data['unet']


def train(run_name, experiments_dir):
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

    # create train and validation generators
    training_batch_generator = create_batch_iterator(user_config=user_config,
                                                     mode='training',
                                                     cpus=train_config['cpus'],
                                                     number_of_batches=train_config['train_batches'])

    validation_batch_generator = create_batch_iterator(mode='validation',
                                                       user_config=user_config,
                                                       cpus=train_config['cpus'],
                                                       number_of_batches=train_config['val_batches'])

    print('\nTraining dataset ')
    print_dataset_statistics(training_batch_generator.dataset)
    print('\nValidation dataset ')
    print_dataset_statistics(validation_batch_generator.dataset)

    # originally defined UNet (with valid convolutions)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=train_config['n_channels'], n_classes=train_config['n_classes']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    min_val = float('inf')

    for n in range(train_config['epochs']):

        tr_losses = []
        val_losses = []

        for idx, (x, y, info) in enumerate(training_batch_generator):

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
            tr_losses.append(loss.item())

        # validate
        with torch.no_grad():
            for idx, (x, y, info) in enumerate(validation_batch_generator):

                # dysplastic vs non-dysplastic
                y = to_dysplastic_vs_non_dysplastic(y)

                # transform x and y
                x = torch.tensor(x.astype('float32'))
                x = torch.transpose(x, 1, 3).to(device)
                y = torch.tensor(y.astype('int64')).to(device)

                # forward and validate
                y_hat = model.forward(x)
                loss = criterion(y_hat, y)
                val_losses.append(loss.item())

        avg_tr_loss = np.mean(tr_losses)
        avg_val_loss = np.mean(val_losses)
        print("Epoch: {}, train loss: {:.3f}, val loss: {:.3f}".format(n, avg_tr_loss, avg_val_loss))

        # save best model
        if avg_val_loss < min_val:
            torch.save(model.state_dict(), os.path.join(exp_dir, 'model_epoch_{}_loss_{:.3f}.pt').format(n, avg_val_loss))
            min_val = avg_val_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='test', help="the name of this experiment")
    parser.add_argument("--exp_dir", type=str, default='/home/mbotros/experiments/barrett_gland_grading',
                        help="experiment dir")
    args = parser.parse_args()
    train(args.run_name, args.exp_dir)
