import copy
from typing import Callable

import equinox as eqx
import jax
from jaxtyping import Array


class MLP(eqx.Module):
    layers: list
    n_units: list
    relu: Callable

    def __init__(self, key, n_units):
        super().__init__()
        n_keys = jax.random.split(key, len(n_units) - 1)
        self.n_units = copy.copy(n_units)

        self.layers = []
        for i in range(1, len(n_units)):
            layer = eqx.nn.Linear(
                in_features=n_units[i - 1],
                out_features=n_units[i],
                use_bias=False,
                key=n_keys[i - 1],
            )
            self.layers.append(layer)

        self.relu = jax.nn.relu

    def __call__(self, x: Array) -> Array:
        x = x.reshape(-1, self.n_units[0])
        out = self.layers[0](x)

        for layer in self.layers[1:]:
            out = self.relu(out)
            out = layer(out)

        return out


def create_mlp(in_size, hidden_sizes, out_size, key):
    model = MLP(key=key, n_units=[in_size] + hidden_sizes + [out_size])
    return model
