"""
The :mod:`deepconnectome.models.dnn` module includes classes and functions for constructing deep neural networks
including BrainNetCNN.
"""

import torch
from torch import nn


class BaseNN(nn.Module):
    """Base Neural Network class that inherits from torch.nn.Module."""

    def __init__(self):
        super().__init__()

    @property
    def _initialize_layers(self):
        """Initializes weights. Fills neural layers with numbers from
        uniform distribution using method shown in Glorot & Bengio, 2010,
        and bias with zeros."""
        for m in self.modules():  # initializing weights
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return self


class Edge2Edge(nn.Module):
    """Edge to Edge block presented in Kawahara et al., 2016.
    It contains cross-shape CNN filters instead of classic box-shape filters.
    In this way, it utilizes topological characteristics of networks."""

    def __init__(self, in_planes, planes, example, bias=False):
        super().__init__()
        self.d = example.size(3)
        self.cnn1 = nn.Conv2d(in_planes, planes, (1, self.d), bias=bias)
        self.cnn2 = nn.Conv2d(in_planes, planes, (self.d, 1), bias=bias)

    def forward(self, x):
        return torch.cat([self.cnn1(x)] * self.d, 3) + torch.cat([self.cnn2(x)] * self.d, 2)


class KawaharaBNCNN(nn.Module):
    """https://doi.org/10.1016/j.neuroimage.2016.09.046"""

    def __init__(self, example, transformed_data):
        print('\nInitializing (Kawahara et al., 2016) BrainNetCNN architecture')
        super().__init__()
        set_attrs_from_parent_instance(self, transformed_data, ['n_classes', 'multiclass', 'n_outcomes'])
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = Edge2Edge(1, 32, example, bias=True)
        self.e2econv2 = Edge2Edge(32, 32, example, bias=True)
        self.E2N = nn.Conv2d(32, 64, (1, self.d))
        self.N2G = nn.Conv2d(64, 256, (self.d, 1))
        self.dense1 = nn.Linear(256, 128)
        self.dense2 = nn.Linear(128, 30)
        self.batchnorm = nn.BatchNorm1d(30)
        self.dense3 = nn.Linear(30, self.n_outcomes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # taken from paper section 2.3 description
    def forward(self, x):
        out = F.leaky_relu(self.e2econv1(x), negative_slope=0.33)
        out = F.leaky_relu(self.e2econv2(out), negative_slope=0.33)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.relu(self.dense1(out))
        out = F.dropout(F.relu(self.dense2(out)), p=0.5)
        out = F.relu(self.dense3(out))
        return out





class HeSexBNCNN(nn.Module):
    """https://dx.doi.org/10.1101/473603"""

    def __init__(self, example, transformed_data):
        print('\nInitializing BNCNN: (He et al. 2018, Sex) BrainNetCNN architecture...')
        super().__init__()
        set_attrs_from_parent_instance(self, transformed_data, ['n_classes'])

        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = Edge2Edge(example.size(1), 38, example, bias=True)
        self.E2N = nn.Conv2d(38, 58, (1, self.d))
        self.N2G = nn.Conv2d(58, 7, (self.d, 1))
        self.dense1 = nn.Linear(7, self.n_classes)

        for m in self.modules():  # initializing weights
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = F.dropout(self.e2econv1(x), p=0.463)
        out = F.dropout(self.E2N(out), p=0.463)
        out = F.dropout(self.N2G(out), p=0.463)
        out = out.view(out.size(0), -1)
        out = torch.sigmoid(self.dense1(out))

        return out


class PervaizBNCNN(nn.Module):
    """https://doi.org/10.1016/j.neuroimage.2020.116604"""

    def __init__(self, example, transformed_data):
        print('\nInitializing (Pervaiz et al. 2020) BrainNetCNN architecture')
        super().__init__()
        set_attrs_from_parent_instance(self, transformed_data, ['n_classes', 'multiclass', 'n_outcomes'])
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = Edge2Edge(example.size(1), 32, example, bias=True)
        self.e2econv2 = Edge2Edge(32, 64, example, bias=True)
        self.E2N = nn.Conv2d(64, 1, (1, self.d))
        self.N2G = nn.Conv2d(1, 256, (self.d, 1))
        self.dense1 = nn.Linear(256, 128)  # init
        self.dense2 = nn.Linear(128, 30)
        if self.multiclass:
            self.dense3 = nn.Linear(30, self.n_classes)
        else:
            self.dense3 = nn.Linear(30, self.n_outcomes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = F.dropout(F.leaky_relu(self.e2econv1(x), negative_slope=0.33), p=.5)
        out = F.dropout(F.leaky_relu(self.e2econv2(out), negative_slope=0.33), p=.5)
        out = F.leaky_relu(self.E2N(out), negative_slope=0.33)
        out = F.dropout(F.leaky_relu(self.N2G(out), negative_slope=0.33), p=0.5)
        out = out.view(out.size(0), -1)
        out = F.dropout(F.relu(self.dense1(out)), p=0.5)
        out = F.dropout(F.relu(self.dense2(out)), p=0.5)

        if self.multiclass:
            out = torch.sigmoid(self.dense3(out))
        else:
            out = F.relu(self.dense3(out))

        return out





class He58behaviorsBNCNN(nn.Module):
    """https://dx.doi.org/10.1101/473603"""

    def __init__(self, example, transformed_data):
        print('\nInitializing BNCNN: (He et al. 2018, 58 behaviors) BrainNetCNN architecture...')
        super().__init__()
        set_attrs_from_parent_instance(self, transformed_data, ['n_classes', 'multiclass', 'n_outcomes'])
        self.in_planes = example.size(1)
        self.d = example.size(3)

        self.e2econv1 = Edge2Edge(example.size(1), 18, example, bias=True)
        self.E2N = nn.Conv2d(18, 19, (1, self.d))
        self.N2G = nn.Conv2d(19, 84, (self.d, 1))

        if self.multiclass:
            self.dense1 = nn.Linear(84, self.n_classes)
        else:
            self.dense1 = nn.Linear(84, self.n_outcomes)

        for m in self.modules():  # initializing weights
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = F.dropout(self.e2econv1(x), p=0.463)
        out = F.dropout(self.E2N(out), p=0.463)
        out = F.dropout(self.N2G(out), p=0.463)
        out = out.view(out.size(0), -1)
        out = self.dense1(out)

        return out

