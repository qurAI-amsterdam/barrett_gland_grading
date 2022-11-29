import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMil(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_classes=3):
        super(DeepMil, self).__init__()
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
        return y_logit.squeeze()


class GatedDeepMil(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_classes=3):
        super(GatedDeepMil, self).__init__()
        self.i = input_size
        self.h = hidden_size
        self.c = num_classes

        self.attention_V = nn.Sequential(
            nn.Linear(self.i, self.h),
            nn.Tanh(),
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.i, self.h),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.h, 1)
        self.classifier = nn.Linear(self.i, self.c)

    def forward(self, x):                             # x:    (B, L, I)
        a_v = self.attention_V(x)                     # a_v:  (B, L, H)
        a_u = self.attention_U(x)                     # a_u:  (B, L, H)
        a = self.attention_weights(a_v * a_u)         # element wise multiplication: (B, L, H) => (B, L, 1)
        a = torch.transpose(a, 1, 2)                  # a: (B, L, 1) => (B, 1, L)
        a = F.softmax(a, dim=1)
        m = torch.matmul(a, x)                        # m: (B, L, I) @ (B, 1, L) => (B, 1, I)
        y_logit = self.classifier(m)
        return y_logit.squeeze()