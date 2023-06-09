# Defines a multi-layer perceptron model and related functions.
#
# Author: Stepp1

import copy
import math

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_units, init_scale=1.0):
        super(MLP, self).__init__()

        self._n_units = copy.copy(n_units)
        self._layers = []
        self.relu = nn.ReLU()

        for i in range(1, len(n_units)):
            layer = nn.Linear(n_units[i - 1], n_units[i], bias=False)

            variance = math.sqrt(2.0 / (n_units[i - 1] + n_units[i]))
            layer.weight.data.normal_(0.0, init_scale * variance)
            self._layers.append(layer)

            name = "fc%d" % i
            if i == len(n_units) - 1:
                name = "fc"  # the prediction layer is just called fc
            self.add_module(name, layer)

    def forward(self, x):
        x = x.view(-1, self._n_units[0])
        out = self._layers[0](x)

        for layer in self._layers[1:]:
            out = self.relu(out)
            out = layer(out)

        return out


def create_mlp(in_size, hidden_sizes, out_size):
    model = MLP([in_size] + hidden_sizes + [out_size])
    return model
