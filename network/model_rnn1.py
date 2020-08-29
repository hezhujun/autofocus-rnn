import torch
import torch.nn as nn
import torch.nn.functional as F

from config import get_config
# from network.resnet import resnet18
from torchvision.models import resnet18, resnet50
from collections import OrderedDict


class MyGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MyGRU, self).__init__()
        self.conv_c_x = nn.Conv2d(input_size, hidden_size, 1)
        self.conv_c_h = nn.Conv2d(input_size, hidden_size, 1)

        self.conv_u_x = nn.Conv2d(input_size, hidden_size, 1)
        self.conv_u_h = nn.Conv2d(input_size, hidden_size, 1)

    def forward(self, x, h):
        c = torch.tanh(self.conv_c_x(x) + self.conv_c_h(h))
        u = torch.sigmoid(self.conv_u_x(x) + self.conv_u_h(h))
        h_next = u * c + (-u + 1) * h
        return h_next


class Head(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super(Head, self).__init__()
        self.gru = nn.GRUCell(in_channels, hidden_dim)
        self.fn1 = nn.Linear(hidden_dim, hidden_dim)
        self.reg = nn.Linear(hidden_dim, 1)

    def forward(self, x, h):
        # x: (BS, C, 1, 1)
        # h: (BS, C, 1, 1)
        y = self.gru(x, h)

        y = self.fn1(y)
        y = torch.tanh(y)

        y = self.reg(y)
        return y


class MyModule(nn.Module):

    def __init__(self, a_dim, feature_type, feature_len=0, freeze_bn=False, freeze_bn_affine=False):
        super(MyModule, self).__init__()
        self.freeze_bn = freeze_bn
        self.freeze_bn_affine = freeze_bn_affine
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
            self.cnn = model
        elif feature_type == "focus_measures":
            x_dim = feature_len
        assert x_dim is not None
        self.x_dim = x_dim
        self.head = Head(x_dim, a_dim)
        self.head1 = Head(x_dim, a_dim)
        self.head2 = Head(x_dim, a_dim)
        self.head3 = Head(x_dim, a_dim)
        self.head4 = Head(x_dim, a_dim)
        self.heads = [self.head, self.head1, self.head2, self.head3, self.head4]
        # self.tanhHead = TanhHead(x_dim)
        # self.heads = [self.tanhHead, self.head1, self.head2, self.head3, self.head4]
        # self.model0 = _Model()
        # self.model1 = _Model()
        # self.model2 = _Model()
        # self.model3 = _Model()
        # self.model4 = _Model()
        # self.models = [self.model0, self.model1, self.model2, self.model3, self.model4]

    def forward(self, x, h, i=0):
        # print("model forward i={}".format(i))
        if self.feature_type == "cnn_features":
            x = self.cnn(x)
            x = torch.flatten(x, start_dim=1)
        y = self.heads[i](x, h)
        y = y.squeeze(dim=1)
        return x, y

    def train(self, mode=True):
        super(MyModule, self).train(mode)
        if self.freeze_bn:
            print("freeze bn params")
            for m in self.cnn.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn_affine:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

            config = get_config()
            for i in range(config["iter_len"] - 1):
                for m in self.heads[i].modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        if self.freeze_bn_affine:
                            m.weight.requires_grad = False
                            m.bias.requires_grad = False
