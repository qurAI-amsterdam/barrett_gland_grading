from distutils.dir_util import copy_tree
import shutil
import torch
import torch.nn as nn
import numpy as np
from wholeslidedata.iterators import create_batch_iterator
from preprocessing import to_dysplastic_vs_non_dysplastic
import os
import argparse
from utils import print_dataset_statistics, plot_pred_batch, plot_confusion_matrix
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import wandb
from utils import mean_metrics, load_config
import segmentation_models_pytorch as smp
from preprocessing import get_preprocessing, tissue_mask_batch


def train(run_name, experiments_dir, wandb_key):
    """Training script for gland grading.

    Args:
        run_name: the name of this experiments run.
        experiments_dir: the directory where to store a copy of the source code, weights and intermediate results.
        wandb_key: weights and biases key for remote logging.
    """
    # config path
    base_dir = '/home/mbotros/code/barrett_gland_grading/'
    user_config = os.path.join(base_dir, 'configs/base_config.yml')

    # make experiment dir & copy source files (config and training script)
    exp_dir = os.path.join(experiments_dir, run_name)
    print('\nExperiment stored at: {}'.format(exp_dir))
    copy_tree(os.path.join(base_dir, 'configs'), os.path.join(exp_dir, 'src', 'configs'))
    copy_tree(os.path.join(base_dir, 'nn_archs'), os.path.join(exp_dir, 'src', 'nn_archs'))
    shutil.copy2(os.path.join(base_dir, 'train.py'), os.path.join(exp_dir, 'src'))

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
    wandb.init(project="Barrett's Gland Grading NDvsD",
               dir=exp_dir,
               config={'data': {'sampling': wholeslide_config,
                                'train data': train_data_dict,
                                'validation data': val_data_dict},
                                'train_config': train_config})
    wandb.run.name = run_name
    print('')

    # create model and put on device(s)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # UNet with ResNet34 encoder
    model = smp.Unet(
        encoder_name=train_config['encoder_name'],        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_depth=train_config['encoder_depth'],      # number of stages used in encoder
        encoder_weights=train_config['encoder_weights'],  # use `imagenet` pretrained weights for encoder initialization
        in_channels=train_config['n_channels'],           # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=train_config['n_classes'],                # model output channels (number of classes in your dataset)
        activation=None,                                  # return logits
    )

    # apply specific preprocessing when using pretrained weights
    if train_config['encoder_weights']:
        preprocessing = get_preprocessing(smp.encoders.get_preprocessing_fn(
            train_config['encoder_name'], train_config['encoder_weights']))
    else:
        preprocessing = get_preprocessing()

    model = nn.DataParallel(model) if train_config['gpus'] > 1 else model
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=train_config['scheduler_step'],
                                                gamma=train_config['scheduler_gamma'])
    criterion = nn.CrossEntropyLoss()
    min_val = float('inf')

    for n in range(train_config['epochs']):

        train_metrics = {}
        validation_metrics = {}

        for idx in tqdm(range(train_config['train_batches']), desc='Epoch {}, lr: {}'.format(n + 1, scheduler.get_last_lr()[0])):
            x_np, y_np, info = next(training_batch_generator)

            # tissue masking and preprocessing
            y_np = tissue_mask_batch(x_np, y_np)
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

                # tissue masking and preprocessing
                y_np = tissue_mask_batch(x_np, y_np)
                sample = preprocessing(image=x_np, mask=y_np)
                x, y = sample['image'].to(device), sample['mask'].to(device)

                # forward and validate
                y_hat = model.forward(x)
                loss = criterion(y_hat, y)

                # store one example: image not normalized but augmented, mask augmented and tissue masked
                example_val_batch_x = x_np
                example_val_batch_y = to_dysplastic_vs_non_dysplastic(y_np)
                example_val_batch_y_hat = y_hat.cpu().detach().numpy()

                # compute dice
                y = y.cpu().detach().numpy().flatten()
                y_hat = torch.argmax(y_hat, dim=1).cpu().detach().numpy().flatten()
                validation_metrics[idx] = {'loss': loss.item(),
                                           'dice per class': f1_score(y, y_hat, average=None, labels=[0, 1, 2]),
                                           'dice weighted': f1_score(y, y_hat, average='weighted'),
                                           'confusion matrix': confusion_matrix(y, y_hat, normalize='true')}

        # accumulate metrics over the epoch
        train_metrics_mean = mean_metrics(train_metrics)
        validation_metrics_mean = mean_metrics(validation_metrics)

        print("Train loss: {:.3f}, val loss: {:.3f}".format(train_metrics_mean['loss'], validation_metrics_mean['loss']))
        print("Train dice: {}, val dice: {}".format(np.round(train_metrics_mean['dice per class'], decimals=2),
                                                    np.round(validation_metrics_mean['dice per class'], decimals=2)))

        # plot predictions on the validation set
        os.makedirs(os.path.join(exp_dir, 'val_predictions'), exist_ok=True)
        pred_save_path = os.path.join(exp_dir, 'val_predictions', 'predictions_epoch_{}.png'.format(n))
        cm_save_path = os.path.join(exp_dir, 'val_predictions', 'confusion_matrix_epoch_{}.png'.format(n))
        plot_pred_batch(example_val_batch_x, example_val_batch_y, example_val_batch_y_hat, save_path=pred_save_path)
        plot_confusion_matrix(validation_metrics_mean['confusion matrix'], save_path=cm_save_path)

        # log metrics, for validation log dices per class
        val_dices = list(validation_metrics_mean['dice per class'])
        wandb.log({'Epoch': n + 1,
                   'train loss': train_metrics_mean['loss'], 'train dice': train_metrics_mean['dice weighted'],
                   'val loss': validation_metrics_mean['loss'], 'val dice': validation_metrics_mean['dice weighted'],
                   'val dice BG': val_dices[0], 'val dice NDBE': val_dices[1], 'val dice DYS': val_dices[2],
                   'prediction': wandb.Image(pred_save_path), 'confusion matrix': wandb.Image(cm_save_path)}
                  )

        # save best model
        os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
        if validation_metrics_mean['loss'] < min_val:
            save_dir = os.path.join(exp_dir, 'checkpoints', 'model_epoch_{}_loss_{:.3f}_dice_{:.3f}.pt'.
                                    format(n, validation_metrics_mean['loss'], validation_metrics_mean['dice weighted']))
            torch.save(model.module.state_dict(), save_dir)
            min_val = validation_metrics_mean['loss']

        # scheduler step
        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='test', help="the name of this experiment")
    parser.add_argument("--exp_dir", type=str, default='/data/archief/AMC-data/Barrett/experiments/barrett_gland_grading/NDvsD',
                        help="experiment dir")
    parser.add_argument("--wandb_key", type=str, help="key for logging to weights and biases")
    args = parser.parse_args()
    train(args.run_name, args.exp_dir, args.wandb_key)
