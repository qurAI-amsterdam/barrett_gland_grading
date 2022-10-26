import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMil(nn.Module):
    def __init__(self, bag_size=25, input_size=4, hidden_size=128, num_classes=3):
        super(DeepMil, self).__init__()
        self.b = bag_size
        self.i = input_size
        self.h = hidden_size
        self.c = num_classes

        self.attention = nn.Sequential(
            nn.Linear(self.i, self.h),
            nn.Tanh(),
            nn.Linear(self.h, 1)
        )
        self.classifier = nn.Linear(self.i, self.c)

    def forward(self, x):
        a = self.attention(x)
        a = torch.transpose(a, 1, 2)
        a = F.softmax(a, dim=1)
        m = torch.matmul(a, x)
        y_logit = self.classifier(m)
        return y_logit
