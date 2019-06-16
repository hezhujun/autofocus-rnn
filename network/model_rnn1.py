import torch
import torch.nn as nn

from network.resnet import resnet18


class MyMode(nn.Module):

    def __init__(self, a_dim, feature_type, feature_len=0):
        super(MyMode, self).__init__()
        c_dim = a_dim
        x_dim = None
        self.feature_type = feature_type
        if feature_type == "cnn_features":
            self.resnet18 = resnet18(pretrained=False)
            x_dim = self.resnet18.fc.in_features

        elif feature_type == "focus_measures":
            x_dim = feature_len
        assert x_dim is not None
        self.c_fn = nn.Linear((x_dim + a_dim), c_dim)
        self.u_fn = nn.Linear((x_dim + a_dim), c_dim)
        self.f_fn = nn.Linear((x_dim + a_dim), c_dim)
        self.o_fn = nn.Linear((x_dim + a_dim), c_dim)
        self.y_fn = nn.Linear(a_dim, 1)

    def forward(self, x, a, c, i=0):
        if self.feature_type == "cnn_features":
            x = self.resnet18(x)
        x_a = torch.cat((x, a), dim=1)
        c_ = torch.tanh(self.c_fn(x_a))
        u = torch.sigmoid(self.u_fn(x_a))
        f = torch.sigmoid(self.f_fn(x_a))
        o = torch.sigmoid(self.o_fn(x_a))
        c = u * c_ + f * c
        a = o * torch.tanh(c)
        y = self.y_fn(a)
        y = y.view(-1)
        return a, c, y
