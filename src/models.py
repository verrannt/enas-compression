from torch import nn
from collections import OrderedDict

def getSimpleNeuralNet(dim_num=784, n_hidden=256, n_classes=10):
    """
    Return a simple PyTorch Sequential Neural Network.
    """
    return nn.Sequential(
        OrderedDict(
            [
                ("flatten", nn.Flatten()),
                ("layer_1_linear", nn.Linear(dim_num, n_hidden)),
                ("layer_1_act", nn.ReLU()),
                # ("layer_1_dropout", nn.Dropout(.1)),
                ("layer_2_linear", nn.Linear(n_hidden, n_hidden)),
                ("layer_2_act", nn.ReLU()),
                # ("layer_2_dropout", nn.Dropout(.1)),
                ("output", nn.Linear(n_hidden, n_classes)),
            ]
        )
    )