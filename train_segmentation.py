from distutils.dir_util import copy_tree
import shutil
import torch
import torch.nn as nn
import numpy as np
from wholeslidedata.iterators import create_batch_iterator
import os
import argparse
from utils import print_dataset_statistics, plot_pred_batch, plot_confusion_matrix
from tqdm import tqdm
from sklearn.metrics import f1_score, confusion_matrix
import wandb
from utils import mean_metrics, load_config
import segmentation_models_pytorch as smp
from preprocessing import get_preprocessing, tissue_mask_batch


def load_segmentation_model(train_config, activation=None):
    """ Loads the segmentation model.

    Args:
        train_config: config that specifies which model to train.
        activation: activation applied at the end

    """
    print('Loading model: {}, weights: {}'.format(train_config['segmentation_model'], train_config['encoder_weights']))

    if train_config['segmentation_model'] == 'unet++':
        model = smp.UnetPlusPlus(
            encoder_name=train_config['encoder_name'],
            encoder_weights=train_config['encoder_weights'],
            in_channels=train_config['n_channels'],
            classes=train_config['n_classes'],
            activation=activation
        )
    elif train_config['segmentation_model'] == 'deeplabv3+':
        model = smp.DeepLabV3Plus(
            encoder_name=train_config['encoder_name'],
            encoder_weights=train_config['encoder_weights'],
            in_channels=train_config['n_channels'],
            classes=train_config['n_classes'],
            activation=activation
        )
    else:
        model = smp.Unet(
            encoder_name=train_config['encoder_name'],
            encoder_weights=train_config['encoder_weights'],
            in_channels=train_config['n_channels'],
            classes=train_config['n_classes'],
            activation=activation
        )

    return model


def train(run_name, exp_dir, wandb_key):
    """Training script for gland grading.

    Args:
        run_name: the name of this experiments run.
        exp_dir: the directory where to store the weights and intermediate results.
        wandb_key: weights and biases key for remote logging.
    """
    # config path
    user_config = os.path.join(exp_dir, 'user_config.yml')

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
    wandb.init(project="Barrett's Gland Grading 3 Classes",
               dir=exp_dir,
               config={'data': {'sampling': wholeslide_config,
                                'train data': train_data_dict,
                                'validation data': val_data_dict},
                                'train_config': train_config})
    wandb.run.name = run_name
    print('')

    # get a segmentation model, put on device
    model = load_segmentation_model(train_config)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = nn.DataParallel(model) if train_config['gpus'] > 1 else model
    model.to(device)

    # apply specific preprocessing when using pretrained weights
    if train_config['encoder_weights']:
        print('Applying preprocessing for {} with {}'.format(train_config['encoder_name'], train_config['encoder_weights']))
        preprocessing = get_preprocessing(smp.encoders.get_preprocessing_fn(
            train_config['encoder_name'], train_config['encoder_weights']))
    else:
        preprocessing = get_preprocessing()

    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           factor=train_config['scheduler_factor'],
                                                           patience=train_config['scheduler_patience'],
                                                           verbose=True)

    # Dice Loss
    criterion_dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    best_dice = 0

    for n in range(train_config['epochs']):

        train_metrics = {}
        validation_metrics = {}

        for idx in tqdm(range(train_config['train_batches']),
                        desc='Epoch {}, lr: {}'.format(n + 1, optimizer.param_groups[0]['lr'])):
            x_np, y_np, info = next(training_batch_generator)

            # tissue masking and preprocessing
            y_np = tissue_mask_batch(x_np, y_np)
            sample = preprocessing(image=x_np, mask=y_np)
            x, y = sample['image'].to(device), sample['mask'].to(device)

            # forward and update
            optimizer.zero_grad()
            y_hat = model.forward(x)
            train_loss = criterion_dice(y_hat, y)
            train_loss.backward()
            optimizer.step()

            # compute and store metrics
            y = y.cpu().detach().numpy().flatten()
            y_hat = torch.argmax(y_hat, dim=1).cpu().detach().numpy().flatten()
            train_metrics[idx] = {'loss': train_loss.item(),
                                  'dice per class': f1_score(y, y_hat, average=None, labels=list(range(train_config['n_classes']))),
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
                val_loss = criterion_dice(y_hat, y)

                # store one example: image not normalized but augmented, mask augmented and tissue masked
                example_val_batch_x = x_np
                example_val_batch_y = y_np
                example_val_batch_y_hat = y_hat.cpu().detach().numpy()

                # compute dice
                y = y.cpu().detach().numpy().flatten()
                y_hat = torch.argmax(y_hat, dim=1).cpu().detach().numpy().flatten()
                validation_metrics[idx] = {'loss': val_loss.item(),
                                           'dice per class': f1_score(y, y_hat, average=None, labels=list(range(train_config['n_classes']))),
                                           'dice weighted': f1_score(y, y_hat, average='weighted'),
                                           'confusion matrix': confusion_matrix(y, y_hat, normalize='true')}

        # accumulate metrics over the epoch
        train_metrics_mean = mean_metrics(train_metrics)
        validation_metrics_mean = mean_metrics(validation_metrics)

        print("Train loss: {}, val loss: {}".format(train_metrics_mean['loss'], validation_metrics_mean['loss']))
        print("Train dice: {}, val dice: {}".format(np.round(train_metrics_mean['dice per class'], decimals=2),
                                                    np.round(validation_metrics_mean['dice per class'], decimals=2)))

        # plot predictions on the validation set
        os.makedirs(os.path.join(exp_dir, 'val_predictions'), exist_ok=True)
        pred_save_path = os.path.join(exp_dir, 'val_predictions', 'predictions_epoch_{}.png'.format(n + 1))
        cm_save_path = os.path.join(exp_dir, 'val_predictions', 'confusion_matrix_epoch_{}.png'.format(n + 1))
        plot_pred_batch(example_val_batch_x, example_val_batch_y, example_val_batch_y_hat, save_path=pred_save_path)
        plot_confusion_matrix(validation_metrics_mean['confusion matrix'], save_path=cm_save_path)

        # log metrics, for validation log dices per class
        val_dices = list(validation_metrics_mean['dice per class'])
        wandb.log({'Epoch': n + 1,
                   'train loss': train_metrics_mean['loss'], 'train dice': train_metrics_mean['dice weighted'],
                   'val loss': validation_metrics_mean['loss'], 'val dice': validation_metrics_mean['dice weighted'],
                   'val dice BG': val_dices[0], 'val dice NDBE': val_dices[1], 'val dice LGD': val_dices[2],
                   'val dice HGD': val_dices[3],
                   'prediction': wandb.Image(pred_save_path), 'confusion matrix': wandb.Image(cm_save_path)}
                  )

        # save if better dice for all but background
        os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
        dice_no_bg = np.mean(val_dices[1:])

        if dice_no_bg > best_dice:
            best_dice = dice_no_bg
            save_dir = os.path.join(exp_dir, 'checkpoints', 'best_model.pt')
            print('Saving model to: {}.'.format(save_dir))
            torch.save(model.module.state_dict(), save_dir)

        # scheduler step
        scheduler.step(val_loss)

    wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='test', help="the name of this experiment")
    parser.add_argument("--exp_dir", type=str,
                        default='/data/archief/AMC-data/Barrett/experiments/barrett_gland_grading/3_classes',
                        help="experiment dir")
    parser.add_argument("--wandb_key", type=str, help="key for logging to weights and biases")
    args = parser.parse_args()
    train(args.run_name, args.exp_dir, args.wandb_key)
