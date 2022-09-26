import os
import torch
import segmentation_models_pytorch as smp
import yaml
from preprocessing import get_preprocessing


def load_config(user_config):
    with open(user_config, 'r') as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)

    return data['wholeslidedata'], data['net']


def load_segmentation_model(train_config, activation=None):
    """ Loads the segmentation model.

    Args:
        train_config: config that specifies which model to train.
        activation: activation applied at the end

    """
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


def load_trained_segmentation_model(exp_dir, model_path):
    """ Loads the trained model.

    Args:
        exp_dir: directory that hold all the information from an experiments (src, checkpoints)
        model_path: path to the trained model

    """
    user_config = os.path.join(exp_dir, 'base_config.yml')
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
