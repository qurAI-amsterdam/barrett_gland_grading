import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):
    """
    GRU Model for classification on slide level.
    """

    def __init__(self, input_size=4, hidden_size=512, num_layers=1, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.mlp_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        Args:
            x: a sequence of features derived from a patch with shape (B, S, D)

        Returns:
            y_hat: a batch of predictions (=logits) with shape (B, 1)

        """
        _, hidden = self.gru(x)
        output = self.mlp_head(hidden).squeeze()
        return output