import copy

import equinox as eqx
import jax
from jaxtyping import Array, Float


class MLP(eqx.Module):
    layers: list

    def __init__(self, key, n_units):
        n_keys = jax.random.split(key, len(n_units) - 1)
        self._n_units = copy.copy(n_units)

        self.layers = []
        for i in range(1, len(n_units)):
            layer = eqx.nn.Linear(
                n_keys[i - 1], n_units[i - 1], n_units[i], bias=False, key=n_keys[i - 1]
            )
            self.layers.append(layer)

        self.relu = jax.nn.relu

    def __call__(self, x: Float[Array, any]) -> Float[Array, any]:
        x = x.reshape(-1, self._n_units[0])
        out = self.layers[0](x)

        for layer in self.layers[1:]:
            out = self.relu(out)
            out = layer(out)

        return out


def create_mlp(in_size, hidden_sizes, out_size):
    model = MLP([in_size] + hidden_sizes + [out_size])
    return model
