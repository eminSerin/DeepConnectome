"""
The :mod:`deepconnectome.models.dnn` module includes classes and functions for constructing deep neural networks
including BrainNetCNN, FNN and GraphCNN.
"""
import collections
import torch
from torch import nn
import torch.nn.functional as F
from deepconnectome.models.utils import _Flatten


class Edge2Edge(nn.Module):
    """Implementation of the Edge-to-Edge layer presented in Kawahara et al., 2016.
    It contains cross-shape CNN filters instead of classic box-shape filters.
    In this way, it utilizes topological characteristics of networks.

    Parameters:
        in_channels: int
            the number of input channels (e.g. 3)
        filters: int
            the number of output channels (e.g. 64)
        dim: int
            the dimension (length) of cross-shape filter
            (i.e. number of nodes in the adjacency matrix).
    """

    def __init__(self, in_channels, filters, dim):
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.dim = dim
        self.conv_row = nn.Conv2d(self.in_channels, self.filters, (1, self.dim))
        self.conv_col = nn.Conv2d(self.in_channels, self.filters, (self.dim, 1))

    def forward(self, x):
        n_obs = x.shape[0]
        return self.conv_row.expand(n_obs, self.filters, self.dim, self.dim) + \
               self.conv_col.expand(n_obs, self.filters, self.dim, self.dim)


class BaseDNN(nn.Module):
    """Base Deep Neural Network class that inherits from torch.nn.Module"""
    def __init__(self):
        super().__init__()

    def _initialize_weights(self):
        """Initializes weights for each layer."""
        def init_weight(layer):
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, nn.BatchNorm1d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)
            return layer

        for s in self.children():  # initializing weights
            if isinstance(s, nn.Sequential):
                for l in s.children():
                    if isinstance(l, Edge2Edge):
                        for c in l.children():
                            init_weight(c)
                    else:
                        init_weight(l)

    def _add_util_layers(self, layer_list, layer_num, filters, activation_function=None,
                        leaky_alpha=0.33, dropout=False, dropout_rate=0.5,
                        batchnorm=False):
        """Adds activation function, drop-out and batch normalization layers
        if desired.

        Parameters:
            layer_list:
            layer_num:
            filters:
            activation_function:
            leaky_alpha:
            dropout:
            dropout_rate:
            batchnorm:
        """
        if activation_function or activation_function is None:
            if activation_function is None:
                act_fun = nn.LeakyReLU(negative_slope=leaky_alpha)
            layer_list.append((f"actFunc{layer_num}", act_fun))
        if dropout:
            layer_list.append((f"dropOut{layer_num}", nn.Dropout(p=dropout_rate)))
        if batchnorm:
            layer_list.append((f"batchNorm{layer_num}", nn.BatchNorm1d(filters)))


class BaseBrainNetCNN(BaseDNN):
    """Base Neural Network class that inherits from the BaseDNN class."""

    def __init__(self, n_nodes, in_channels, n_outputs=1, classification=False,
                 e2e_layers=2, e2e_filters=32, e2n_filters=64, n2g_filters=256,
                 dense_sml=False, dense_layers=1, dropout_rate=0.5, leaky_alpha=0.33):
        """ Initializes class.

        Parameters:
            n_nodes: int
                the number of ROI regions (e.g. 268)
            in_channels: int
                the number of different connectivity matrices (i.e. different tasks such as resting state,
                working memory).
            n_outputs: int
                the number of outputs. For example, set it to 2 if you want to predict 2 different cognitive
                measures such as personality traits of extraversion and neuroticism scores.
            classification: bool
                if task is classification.
            e2e_layers: int
                the number of Edge-to-Edge layers.
            e2e_filters: int
                the number of Edge-to-Edge filters.
            e2n_filters: int
                the number of Edge-to-Node filters.
            n2g_filters: int
                the number of Node-to-Graph filters.
            dense_sml: bool
                whether drop FC layers; thus have only an output layer after the Node-to-Graph layer.
            dense_layers: int
                the number of dense layers. It increases only the FC128 layer
                while keeping other FC (e.g. FC30) layers intact.
            dropout_rate: float
                the dropout rate.
            leaky_alpha: float
                the leaky_alpha rate.
        """
        super().__init__()
        self.n_nodes = n_nodes
        self.in_channels = in_channels
        self.n_outputs = n_outputs
        self.classification = classification
        self.e2e_layers = e2e_layers
        self.e2e_filters = e2e_filters
        self.e2n_filters = e2n_filters
        self.n2g_filters = n2g_filters
        self.dense_sml = dense_sml
        self.dense_layers = dense_layers
        self.dropout_rate = dropout_rate
        self.leaky_alpha = leaky_alpha
        self._dense_init_filter = 128
        self._dense_last_filter = 30

        # Construct BrainNetCNN architecture
        self.e2e = nn.Sequential(collections.OrderedDict(self._create_e2e()))
        self.e2n = nn.Sequential(collections.OrderedDict(self._create_e2n()))
        self.n2g = nn.Sequential(collections.OrderedDict(self._create_n2g()))
        self.dense = nn.Sequential(collections.OrderedDict(self._create_dense()))
        self._initialize_weights()

    def _create_e2e(self, dropout=False, activation_function=None, batchnorm=False):
        """Creates variable number of Edge-to-Edge layers."""
        for i in range(1, self.e2e_layers+1):
            if i == 1:
                e2e_list = [("e2e1", Edge2Edge(self.in_channels, self.e2e_filters, self.n_nodes))]
            else:
                e2e_list.append((f"e2e{i}", Edge2Edge(self.e2e_filters, self.e2e_filters, self.n_nodes)))
            self._add_util_layers(e2e_list, i, self.e2e_filters, leaky_alpha=self.leaky_alpha, dropout=dropout,
                                  activation_function=activation_function,
                                  dropout_rate=self.dropout_rate, batchnorm=batchnorm)
        return e2e_list

    def _create_e2n(self, dropout=True, activation_function=None, batchnorm=False):
        """Creates Edge-to-Node layer."""
        e2n = [("e2n", nn.Conv2d(self.e2e_filters, self.e2n_filters, (1, self.n_nodes)))]
        self._add_util_layers(e2n, 1, self.e2n_filters, leaky_alpha=self.leaky_alpha,
                        dropout=dropout, dropout_rate=self.dropout_rate,
                        activation_function=activation_function, batchnorm=batchnorm)
        return e2n

    def _create_n2g(self, dropout=False, activation_function=None, batchnorm=False):
        """Creates Node-to-Graph layer."""
        n2g = [("n2g", nn.Conv2d(self.e2n_filters, self.n2g_filters, (self.n_nodes, 1)))]
        self._add_util_layers(n2g, 1, self.n2g_filters, leaky_alpha=self.leaky_alpha,
                        dropout=dropout, dropout_rate=self.dropout_rate,
                        activation_function=activation_function, batchnorm=batchnorm)
        n2g.append(("flatten", _Flatten()))
        return n2g

    def _create_dense(self, dropout=True, activation_function=None, batchnorm=True):
        """Creates variable number of dense layers. Note that, this function
        only increases the number of FC128 layers, while keeping the other dense layers
        such as FC30 layer intact."""
        if not self.dense_sml:
            for i in range(1, 1 + self.dense_layers):
                if i == 1:
                    dense_list = [(f"dense{i}", nn.Linear(self.n2g_filters, self._dense_init_filter))]
                else:
                    dense_list.append((f"dense{i}", nn.Linear(self._dense_init_filter, self._dense_init_filter)))
                self._add_util_layers(dense_list, i, self._dense_init_filter, leaky_alpha=self.leaky_alpha,
                                dropout=dropout, dropout_rate=self.dropout_rate,
                                activation_function=activation_function, batchnorm=batchnorm)

            dense_list.append((f"dense{1 + self.dense_layers}", nn.Linear(self._dense_init_filter,
                                                                          self._dense_last_filter)))
            self._add_util_layers(dense_list, 1 + self.dense_layers, self._dense_last_filter, leaky_alpha=self.leaky_alpha,
                            dropout=dropout, dropout_rate=self.dropout_rate,
                            activation_function=activation_function, batchnorm=batchnorm)
            dense_list.append((f"dense{2 + self.dense_layers}", nn.Linear(self._dense_last_filter, self.n_outputs)))
        else:
            dense_list = [("dense", nn.Linear(self.n2g_filters, self.n_outputs))]
        return dense_list

    def forward(self, x):
        out = self.e2e(x)
        out = self.e2n(out)
        out = self.n2g(out)
        out = self.dense(out)
        if self.classification:
            return F.softmax(out)
        return out


class KawaharaBNCNN(BaseBrainNetCNN):
    """The implementation of the BrainNetCNN architecture presented in
    Kawahara et al., 2016 (https://doi.org/10.1016/j.neuroimage.2016.09.046)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(n_outputs=1, classification=False, e2e_layers=2, e2e_filters=32, e2n_filters=64,
                         n2g_filters=256, dense_sml=False, dense_layers=1, dropout_rate=0.5, leaky_alpha=0.33, **kwargs)
        print('\nInitializing (Kawahara et al., 2016) BrainNetCNN architecture')


class PervaizBNCNN(BaseBrainNetCNN):
    """ The implementation of the BrainNetCNN architecture shown in
    Pervaiz et al., 2020 (https://doi.org/10.1016/j.neuroimage.2020.116604).
    The architecture is presented in the supplementary material (see A.13) of the paper."""

    def __init__(self, *args, **kwargs):
        super().__init__(e2e_layers=1, e2e_filters=256, e2n_filters=128, n2g_filters=256, dense_sml=False,
                         dense_layers=1, dropout_rate=0.5, leaky_alpha=0.01, *args, **kwargs)
        print('\n Initializing (Pervaiz et al. 2020) BrainNetCNN architecture')
        self.e2n = nn.Sequential(collections.OrderedDict(self._create_e2n(dropout=False)))
        self.dense = nn.Sequential(collections.OrderedDict(self._create_dense(activation_function=nn.ReLU())))


class BaseHeBNCNN(BaseBrainNetCNN):
    """The base class for the BrainNetCNN architecture presented in He et al., 2019
    (https://dx.doi.org/10.1101/473603)."""

    def __init__(self,  *args, **kwargs):
        super().__init__(e2e_layers=1, dense_sml=True, dropout_rate=0.463, *args, **kwargs)
        self.e2e = nn.Sequential(collections.OrderedDict(self._create_e2e(dropout=True, activation_function=False)))
        self.e2n = nn.Sequential(collections.OrderedDict(self._create_e2n(dropout=True, activation_function=False)))
        self.n2g = nn.Sequential(collections.OrderedDict(self._create_n2g(dropout=True, activation_function=False)))


class HeSexBNCNN(BaseBrainNetCNN):
    """The implementation of the BrainNetCNN architecture presented in
    He et al., 2019 (https://dx.doi.org/10.1101/473603) to predict gender of
    subjects in the UK Biobank dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(e2e_filters=38, e2n_filters=58, classification=True, n2g_filters=7, *args, **kwargs)
        print('\nInitializing BNCNN: (He et al. 2018, UKB Sex) BrainNetCNN architecture...')


class He58behaviorsBNCNN(BaseBrainNetCNN):
    """The implementation of the BrainNetCNN architecture presented in
    He et al., 2019 (https://dx.doi.org/10.1101/473603) to predict 58
    behavioral measures collected from the subjects from the HCP dataset.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(e2e_filters=18, e2n_filters=19, n2g_filters=84, *args, **kwargs)
        print('\nInitializing BNCNN: (He et al. 2018, HCP 58 behaviors) BrainNetCNN architecture...')


class HeAgeBNCNN(BaseBrainNetCNN):
    """The implementation of the BrainNetCNN architecture presented in
    He et al., 2019 (https://dx.doi.org/10.1101/473603) to predict
    subjects' age from their connectomes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(e2e_filters=22, e2n_filters=79, n2g_filters=91, *args, **kwargs)
        print('\nInitializing BNCNN: (He et al. 2018, UKB Age) BrainNetCNN architecture...')


class HeFluidIntBNCNN(BaseBrainNetCNN):
    """The implementation of the BrainNetCNN architecture presented in
    He et al., 2019 (https://dx.doi.org/10.1101/473603) to predict
    subjects' fluid intelligence scores from their connectomes.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(e2e_filters=40, e2n_filters=60, n2g_filters=41, *args, **kwargs)
        print('\nInitializing BNCNN: (He et al. 2018, UKB Fluid Intelligence) BrainNetCNN architecture...')

class FNN(BaseDNN):
    """The implementation of the Feedforward Neural Network."""

    def __init__(self, in_features, n_outputs=1, classification=False,
                 hidden_layers=3, layer_size=(256, 128, 64, 32), dropout_rate=0.5,
                 leaky_alpha=0.33):
        """ Initializes class.

        Parameters:
            in_features: int
                the number input features.
            n_outputs: int
                the number of outputs.
            classification: bool
                if classification to be done.
            hidden_layers: int
                the number of hidden layers.
            layer_size: tuple, int
                the number of nodes in each layer excluding output layer. the input can be tuple or integer.
                If integer is provided, all hidden layers will have the same number of nodes.
                Of note, if you want to provide tuple, make sure that it also includes number of
                nodes for the fist layer (input), thus the length of tuple must be 1 item
                greater than the number of hidden layers.
            dropout_rate: float
                the dropout rate.
            leaky_alpha: float
                the alpha rate for the leaky ReLu activation function.
        """
        super().__init__()
        print('\nInitializing Feedforward Neural Network architecture...')
        self.in_features = in_features
        self.n_outputs = n_outputs
        self.classification = classification
        self.hidden_layers = hidden_layers
        self.layer_size = layer_size
        self.dropout_rate = dropout_rate
        self.leaky_alpha = leaky_alpha
        self._dropout = False
        self._batchnorm = True

        # Construct feedforward neural network.
        self.fnn = self._create_fnn()
        self._initialize_weights()

    def _create_fnn(self):
        if type(self.layer_size) == int:
            layer_size = (self.layer_size,) * self.hidden_layers+1
        else:
            layer_size = self.layer_size

        for i in range(1, 2 + self.hidden_layers):
            if i == 1:
                fnn_list = [("input_layer", nn.Linear(self.in_features, layer_size[i-1]))]
            else:
                fnn_list.append((f"hidden{i-1}", nn.Linear(layer_size[i-2], layer_size[i-1])))
            self._add_util_layers(fnn_list, i, layer_size[i-1], leaky_alpha=self.leaky_alpha,
                            dropout=self._dropout, dropout_rate=self.dropout_rate,
                            batchnorm=self._batchnorm)

        fnn_list.append((f"output", nn.Linear(layer_size[i-1], self.n_outputs)))

        return nn.Sequential(collections.OrderedDict(fnn_list))

    def forward(self, x):
        if self.classification:
            return F.softmax(self.fnn(x))
        return self.fnn(x)


class FC90(FNN):
    """The implementation of FC90 architecture shown in Kawahara et al., 2016."""
    def __init__(self, *args, **kwargs):
        super().__init__(hidden_layers=1, layer_size=(90, 30), *args, **kwargs)


class FC30(FNN):
    """Implementation of FC30 architecture shown in Kawahara et al., 2016."""
    def __init__(self, *args, **kwargs):
        super().__init__(hidden_layers=0, layer_size=30, *args, **kwargs)


class RCNN(BaseDNN):
    """The implementation of the Recurrent Convulational Neural Network
    architecture shown in Pervaiz et al., 2020."""
    # TODO: Implement Recurrent Convulational Networks.
    def __init__(self):
        super().__init__()
        raise NotImplementedError


class GraphCNN(nn.Module):
    """The implementation of the Graph Convulation Neural Network."""
    # TODO: Implement GraphCNN!
    def __init__(self):
        super().__init__()
        raise NotImplementedError