import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


class SetTransformer(nn.Module):
    def __init__(self, set_size=32, num_outputs=1, dim_out=4, num_inds=32,
                 hidden_dim=128, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(set_size, hidden_dim, num_heads, num_inds, ln=ln),
            ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            nn.Dropout(0.5),
            PMA(hidden_dim, num_heads, num_outputs, ln=ln),
            SAB(hidden_dim, hidden_dim, num_heads, ln=ln),
            SAB(hidden_dim, hidden_dim, num_heads, ln=ln),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.LeakyReLU(0.3), nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, dim_out)
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x).squeeze(-1)
        return x


class DeepSetTransformer(nn.Module):
    def __init__(self, input_dim=512, set_size=32, num_outputs=1, dim_out=4, num_inds=32,
                 hidden_dim=128, num_heads=4, ln=False):
        super(DeepSetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(set_size, hidden_dim, num_heads, num_inds, ln=ln),
            ISAB(hidden_dim, hidden_dim, num_heads, num_inds, ln=ln),
            nn.Dropout(0.5)
        )
        self.dec = nn.Sequential(
            PMA(hidden_dim, num_heads, num_outputs, ln=ln),
            SAB(hidden_dim, hidden_dim, num_heads, ln=ln),
            SAB(hidden_dim, hidden_dim, num_heads, ln=ln),
            nn.Dropout(0.5)
        )

        self.deepset_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(), nn.Dropout(0.3)
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.LeakyReLU(0.3), nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, dim_out)
        )

    def forward(self, x):
        transformer_x = self.enc(x)
        transformer_x = self.dec(transformer_x).squeeze(1)
        deepset_x = self.deepset_encoder(x).mean(dim=1)
        x = torch.cat([transformer_x, deepset_x], dim=1)
        x = self.head(x).squeeze(-1)
        return x


class SmallSetTransformer(nn.Module):
    def __init__(self, set_size=32, hidden_dim=64, dim_out=64, num_heads=4):
        super().__init__()
        print("Set Transformer Model Initialized")
        self.enc = nn.Sequential(
            SAB(dim_in=set_size, dim_out=hidden_dim, num_heads=num_heads),
            SAB(dim_in=hidden_dim, dim_out=dim_out, num_heads=num_heads),
        )
        self.dec = nn.Sequential(
            PMA(dim=dim_out, num_heads=num_heads, num_seeds=1),
            nn.Linear(in_features=dim_out, out_features=4),
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x.squeeze(-1)


class DeepSet(nn.Module):
    def __init__(self, dim_input, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.dim_output = dim_output
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(dim_hidden, dim_hidden))
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(dim_hidden // 2, dim_output))

    def forward(self, X):
        X = self.enc(X).mean(dim=1)
        X = self.dec(X)
        return X
