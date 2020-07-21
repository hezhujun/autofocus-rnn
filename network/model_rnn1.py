import torch
import torch.nn as nn

# from network.resnet import resnet18
from torchvision.models import resnet18
from collections import OrderedDict


class MyMode(nn.Module):

    def __init__(self, a_dim, feature_type, feature_len=0):
        super(MyMode, self).__init__()
        x_dim = None
        self.feature_type = feature_type
        if feature_type == "cnn_features":
            net = resnet18(pretrained=True)
            x_dim = net.fc.in_features

            layers = []
            for name, child in net.named_children():
                layers.append((name, child))
            model = nn.Sequential()
            for name, chlid in layers[:-1]:
                model.add_module(name, chlid)
            self.resnet18 = model
        elif feature_type == "focus_measures":
            x_dim = feature_len
        assert x_dim is not None
        self.lstm = nn.LSTMCell(x_dim, a_dim)
        self.y_fn = nn.Linear(a_dim, 1)

    def forward(self, x, h, c, i=0):
        if self.feature_type == "cnn_features":
            x = self.resnet18(x)
            x = torch.flatten(x, 1)
        h, c = self.lstm(x, (h, c))
        y = self.y_fn(h)
        y = y.squeeze(dim=1)
        return h, c, y
