from distutils.dir_util import copy_tree
import shutil
import torch
import torch.nn as nn
import numpy as np
from nn_archs import UNet
from wholeslidedata.iterators import create_batch_iterator
from preprocessing import to_dysplastic_vs_non_dysplastic
import os
import argparse
from utils import print_dataset_statistics, plot_pred_batch
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb
from utils import mean_metrics, load_config
import segmentation_models_pytorch as smp
from preprocessing import get_preprocessing

"""
To-Do:
(1) Add data augmentation: albumentations included in PathologyWholeSlide data.
"""


def train(run_name, experiments_dir, wandb_key):
    # config path
    base_dir = '/home/mbotros/code/barrett_gland_grading/'
    user_config = os.path.join(base_dir, 'configs/unet_training_config.yml')

    # make experiment dir & copy source files (config and training script)
    exp_dir = os.path.join(experiments_dir, run_name)
    print('\nExperiment stored at: {}'.format(exp_dir))
    copy_tree(os.path.join(base_dir, 'configs'), os.path.join(exp_dir, 'src', 'configs'))
    copy_tree(os.path.join(base_dir, 'nn_archs'), os.path.join(exp_dir, 'src', 'nn_archs'))
    shutil.copy2(os.path.join(base_dir, 'train_unet.py'), os.path.join(exp_dir, 'src'))

    # load network config and store in experiment dir
    print('Loaded config: {}'.format(user_config))
    wholeslide_config, train_config = load_config(user_config)

    # create train and validation generators (no reset)
    training_batch_generator = create_batch_iterator(user_config=user_config,
                                                     mode='training',
                                                     cpus=train_config['cpus'])

    validation_batch_generator = create_batch_iterator(mode='validation',
                                                       user_config=user_config,
                                                       cpus=train_config['cpus'])

    print('\nTraining dataset ')
    train_data_dict = print_dataset_statistics(training_batch_generator.dataset)
    print('\nValidation dataset ')
    val_data_dict = print_dataset_statistics(validation_batch_generator.dataset)
    print('')

    # log EVERYTHING with weights and biases
    os.environ["WANDB_API_KEY"] = wandb_key
    wandb.init(project="Barrett's Gland Grading",
               dir=exp_dir,
               config={'data': {'sampling': wholeslide_config,
                                'train data': train_data_dict,
                                'validation data': val_data_dict},
                       'unet_train_config': train_config})
    wandb.run.name = run_name
    print('')

    # create model and put on device(s)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=train_config['n_channels'], n_classes=train_config['n_classes'])
    preprocessing = get_preprocessing()

    # # deeplab with pretrained resnet50 encoder
    # model = smp.DeepLabV3Plus(
    #     encoder_name='resnet50',                    # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    #     encoder_weights='imagenet',                 # use `imagenet` pretrained weights for encoder initialization
    #     in_channels=train_config['n_channels'],     # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    #     classes=train_config['n_classes'],          # model output channels (number of classes in your dataset)
    #     activation=None,                            # return logits
    # )
    #
    # # apply preprocessing for using pretrained weights
    # preprocessing = get_preprocessing(smp.encoders.get_preprocessing_fn('resnet50', 'imagenet'))

    model = nn.DataParallel(model) if train_config['gpus'] > 1 else model
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    criterion = nn.CrossEntropyLoss()

    min_val = float('inf')

    for n in range(train_config['epochs']):

        train_metrics = {}
        validation_metrics = {}

        for idx in tqdm(range(train_config['train_batches']), desc='Epoch {}'.format(n + 1)):
            x_np, y_np, info = next(training_batch_generator)

            # preprocessing
            sample = preprocessing(image=x_np, mask=y_np)
            x, y = sample['image'].to(device), sample['mask'].to(device)

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
                                  'dice per class': f1_score(y, y_hat, average=None, labels=[0, 1, 2]),
                                  'dice weighted': f1_score(y, y_hat, average='weighted')}

        # validate
        with torch.no_grad():
            for idx in tqdm(range(train_config['val_batches']), desc='Validating'):
                x_np, y_np, info = next(validation_batch_generator)

                # preprocessing
                sample = preprocessing(image=x_np, mask=y_np)
                x, y = sample['image'].to(device), sample['mask'].to(device)

                # forward and validate
                y_hat = model.forward(x)
                loss = criterion(y_hat, y)

                # store one example, not preprocessed
                example_val_batch_x = x_np
                example_val_batch_y = to_dysplastic_vs_non_dysplastic(y_np)
                example_val_batch_y_hat = y_hat.cpu().detach().numpy()

                # compute dice
                y = y.cpu().detach().numpy().flatten()
                y_hat = torch.argmax(y_hat, dim=1).cpu().detach().numpy().flatten()
                validation_metrics[idx] = {'loss': loss.item(),
                                           'dice per class': f1_score(y, y_hat, average=None, labels=[0, 1, 2]),
                                           'dice weighted': f1_score(y, y_hat, average='weighted')}

        # compute and print metrics
        training_means = mean_metrics(train_metrics)
        validation_means = mean_metrics(validation_metrics)
        print("Train loss: {:.3f}, val loss: {:.3f}".format(training_means['loss'], validation_means['loss']))
        print("Train dice: {}, val dice: {}".format(np.round(training_means['dice per class'], decimals=2),
                                                    np.round(validation_means['dice per class'], decimals=2)))

        # plot predictions on the validation set
        os.makedirs(os.path.join(exp_dir, 'val_predictions'), exist_ok=True)
        image_save_path = os.path.join(exp_dir, 'val_predictions', 'predictions_epoch_{}.png'.format(n))
        plot_pred_batch(example_val_batch_x, example_val_batch_y, example_val_batch_y_hat, save_path=image_save_path)

        # log metrics, for validation log dices per class
        val_dice_per_class = list(np.round(validation_means['dice per class'], decimals=2))
        wandb.log({'epoch': n + 1,
                   'train loss': training_means['loss'], 'train dice': training_means['dice weighted'],
                   'val loss': validation_means['loss'], 'val dice': validation_means['dice weighted'],
                   'val dice BG': val_dice_per_class[0],
                   'val dice NDBE': val_dice_per_class[1],
                   'val dice DYS': val_dice_per_class[2],
                   'prediction': wandb.Image(image_save_path)})

        # save best model
        os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
        if validation_means['loss'] < min_val:
            save_dir = os.path.join(exp_dir, 'checkpoints', 'model_epoch_{}_loss_{:.3f}_dice_{:.3f}.pt'.
                                    format(n, validation_means['loss'], validation_means['dice weighted']))
            torch.save(model.state_dict(), save_dir)
            min_val = validation_means['loss']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='test', help="the name of this experiment")
    parser.add_argument("--exp_dir", type=str, default='/home/mbotros/experiments/barrett_gland_grading',
                        help="experiment dir")
    parser.add_argument("--wandb_key", type=str, help="key for logging to weights and biases")
    args = parser.parse_args()
    train(args.run_name, args.exp_dir, args.wandb_key)
