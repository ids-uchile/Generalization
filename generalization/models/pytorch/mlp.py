"""
Defines a multi-layer perceptron model and related functions.

Author: Stepp1
"""

import copy

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_units, use_dropout=False, dropout_rate=0.5):
        super(MLP, self).__init__()

        self.n_units = copy.copy(n_units)
        self.layers = []
        self.relu = nn.LeakyReLU()

        for i in range(1, len(n_units)):
            layer = nn.Linear(n_units[i - 1], n_units[i], bias=True)

            self.layers.append(layer)

            name = "fc%d" % i
            if i == len(n_units) - 1:
                name = "fc"  # the prediction layer is just called fc
            self.add_module(name, layer)

        self.dropout = nn.Dropout(dropout_rate) if use_dropout else nn.Identity()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.layers[0](x)

        for i, layer in enumerate(self.layers[1:]):
            out = self.relu(out)
            if i == len(self.layers) - 2:
                out = self.dropout(out)
            out = layer(out)

        return out


def mlp(in_size, hidden_sizes, out_size, **kwargs):
    use_dropout = kwargs.get("use_dropout", False)
    dropout_rate = kwargs.get("dropout_rate", 0.5)

    model = MLP([in_size] + hidden_sizes + [out_size], use_dropout, dropout_rate)
    return model
