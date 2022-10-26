import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, bag_size=25, input_size=4, hidden_size=128, num_classes=3):
        super(Attention, self).__init__()
        self.bag_size = bag_size
        self.input_size = input_size
        self.hidden = hidden_size
        self.num_classes = num_classes

        self.attention = nn.Sequential(
            nn.Linear(self.input_size, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 1),
            nn.Softmax(dim=1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.bag_size * self.K, self.num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x = (B, 25, 4)
        a = self.attention(x)              # (B, 25, 1)
        M = torch.mm(a.T, x)               # (1, 25) @ (25, 4) => (1, 4)
        y_prob = self.classifier(M)

        return y_prob, a
