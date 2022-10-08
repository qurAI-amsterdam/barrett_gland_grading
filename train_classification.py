import numpy as np
import tqdm
import os
import torch
from torch import nn
import argparse


class SlideGradeGRU:
    def __init__(self, input_size=4, hidden_size=128, num_layers=1, num_classes=3):
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.mlp_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        output, hn = self.gru(x)
        return NotImplementedError


def train(run_name, experiments_dir, wandb_key):
    """ Train something attention-based with segmentations outputs
    """


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default='test', help="the name of this experiment")
    parser.add_argument("--exp_dir_classification", type=str,
                        default='/data/archief/AMC-data/Barrett/experiments/barrett_slide_classification/top_25_entropy',
                        help="experiment dir classification")
    parser.add_argument("--wandb_key", type=str, help="key for logging to weights and biases")
    args = parser.parse_args()
    train(args.run_name, args.exp_dir, args.wandb_key)
