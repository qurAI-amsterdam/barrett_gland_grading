import shutil
import torch
import torch.nn as nn
import numpy as np
from nn_archs import UNet
import yaml
from wholeslidedata.iterators import create_batch_iterator
from label_utils import to_dysplastic_vs_non_dysplastic
import os
import argparse

"""
To-do:
(1) add validation
    * make a split on slide level define in data.yml
(2) add logging with weights and biases
(3) ...
"""


def load_config(config):

    with open(config, 'r') as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)

    return data['unet']


def train(run_name, experiments_dir):

    # config paths
    train_config = './configs/unet_training_config.yml'
    data_config = './configs/data.yml'
    print('Loaded config: {}'.format(train_config))

    # make experiment dir & copy the train script and config files
    exp_dir = os.path.join(experiments_dir, run_name)
    os.makedirs(exp_dir, exist_ok=True)
    print('Experiment stored at: {}'.format(exp_dir))
    shutil.copy2(train_config, exp_dir)
    shutil.copy2(data_config, exp_dir)
    shutil.copy2('train_unet.py', exp_dir)

    return 0

    # load network config and store in experiment dir
    config = load_config(train_config)

    # originally defined UNet (with valid convolutions)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=config['n_channels'], n_classes=config['n_classes']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    # create a batch iterator
    with create_batch_iterator(user_config=config,
                               mode='training',
                               cpus=config['cpus']) as training_batch_generator:

        for n in range(config['epochs']):

            tr_losses = []
            val_losses = []

            for idx, (x, y, info) in enumerate(training_batch_generator):

                # dysplastic vs non-dysplastic
                y = to_dysplastic_vs_non_dysplastic(y)

                # transform x and y
                x = torch.tensor(x.astype('float32'))
                x = torch.transpose(x, 1, 3).to(device)
                y = torch.tensor(y.astype('int64')).to(device)

                optimizer.zero_grad()
                y_hat = model.forward(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                tr_losses.append(loss.item())

            print("Epoch: {}, train loss: {}".format(n, np.mean(tr_losses)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='test', help="the name of this experiment")
    parser.add_argument("--exp_dir", type=str, default='/home/mbotros/experiments/barrett_gland_grading',
                        help="experiment dir")
    args = parser.parse_args()
    train(args.run_name, args.exp_dir)

