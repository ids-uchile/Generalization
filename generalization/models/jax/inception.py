from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax
import jaxpruner
import ml_collections
import optax
from flax import linen as nn
from flax.training import train_state
from jax import numpy as jnp


class ConvNormActivation(nn.Module):
    """A simple port of `torchvision.ops.misc.ConvNormActivation`.

    Packs `convolution` -> `normalisation` -> `activation` into one easy to use module.
    - `out_channels`: Number of channels produced by the Convolution-Normalzation-Activation block
    - `kernel_size`: Size of the convolution kernel. Defaults to `3`
    - `stride`: Stride of the convolution. Defaults to `1`
    - `padding`: Padding added to all four sides of the input. Defaults to `SAME`,
    - `groups`: Number of blocked connections from input channels to output channels. Defaults to `1`
    - `norm_layer`: Norm layer that will be stacked on top of the convolution layer. If ``None``
        this layer wont be used. Defaults to ``nn.BatchNorm``
    - `activation_layer`: Activation function which will be stacked on top of the normalization layer
        (if not None), otherwise on top of the conv layer
        If ``None`` this layer wont be used. Defaults to ``jax.nn.relu``
    - `dilation`: Spacing between kernel elements. Defaults to `1`
    - `use_bias`: If `True`, bias is used in the convolution layer. By default, biases are included #
        if ``norm_layer is None``
    - `dtype`: The dtype of the computation (default: float32). This defaults to the dtype of the input
    """

    out_channels: int
    kernel_size: int = 3
    stride: Union[int, Sequence[int]] = 1
    padding: Optional[Union[Sequence[int], str]] = "SAME"
    groups: int = 1
    norm_layer: Any = nn.BatchNorm
    activation: Any = nn.relu
    dilation: int = 1
    groups: int = 1
    use_bias: Optional[bool] = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = jnp.array(x, dtype=self.dtype)
        x = nn.Conv(
            features=self.out_channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=self.stride,
            padding=self.padding,
            # input_dilation=self.dilation,
            kernel_dilation=self.dilation,
            use_bias=self.use_bias,
            # param_dtype=self.dtype,
            dtype=self.dtype,
            name="conv",
        )(x)
        x = self.norm_layer(
            name="bn_norm", epsilon=0.01, dtype=self.dtype, param_dtype=self.dtype
        )(x, use_running_average=not training)
        return self.activation(x)


class ConvModule(nn.Module):
    """
    Basic ConvModule. Wraps `ConvNormActivation` with a few default values.
    """

    out_channels: int
    kernel_size: int
    stride: Union[int, Sequence[int]] = 1
    padding: Optional[Union[Sequence[int], str]] = "SAME"
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = False):
        return ConvNormActivation(
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dtype=self.dtype,
            name="ConvNormActivation",
        )(x, training=training)


class InceptionModule(nn.Module):
    """
    Basic InceptionModule

    Note:
    In the Understanding generalisation paper, padding is 0 and 1 respectively for 1x1 and 3x3 convolutions.
    Here, padding is always SAME.
    """

    out_1x1: int
    out_3x3: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = False):
        out_conv1x1 = ConvModule(
            self.out_1x1, kernel_size=1, stride=1, dtype=self.dtype, padding=0
        )(x, training=training)
        out_conv3x3 = ConvModule(
            self.out_3x3, kernel_size=3, stride=1, dtype=self.dtype, padding=1
        )(x, training=training)

        return jnp.concatenate([out_conv1x1, out_conv3x3], -1)


class DownSampleModule(nn.Module):
    """
    Basic DownSampleModule that uses max pooling.
    """

    out_3x3: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = False):
        out_conv3x3 = ConvModule(
            self.out_3x3, kernel_size=3, stride=2, dtype=self.dtype, padding="VALID"
        )(x, training=training)
        out_pool = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2))
        return jnp.concatenate([out_conv3x3, out_pool], -1)


class InceptionSmall(nn.Module):
    """
    Inception Small as shown in the Appendix A of the paper.

    The implementation follows the blocks from Figure 3.
    """

    num_classes: int = 10
    dropout_rate: float = 0.5
    use_dropout: bool = False
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = jnp.array(x, dtype=self.dtype)
        x = ConvModule(96, kernel_size=3, stride=1, padding=0, dtype=self.dtype)(x)

        # inception x 2 -> downsample1
        x = InceptionModule(32, 32, dtype=self.dtype)(x)  # N,H,W,64
        x = InceptionModule(32, 48, dtype=self.dtype)(x)  # N,H,W,80
        x = DownSampleModule(80, dtype=self.dtype)(x)

        # inception x 4 -> downsample2
        x = InceptionModule(112, 48, dtype=self.dtype)(x)  # N,H,W,160
        x = InceptionModule(96, 64, dtype=self.dtype)(x)  # N,H,W,160
        x = InceptionModule(80, 80, dtype=self.dtype)(x)  # N,H,W,160
        x = InceptionModule(48, 96, dtype=self.dtype)(x)  # N,H,W,160
        x = DownSampleModule(96, dtype=self.dtype)(x)

        # inception x 2 -> AdaptiveAvgPool2d -> fc
        x = InceptionModule(176, 160, dtype=self.dtype)(x)  # N,H,W,336
        x = InceptionModule(176, 160, dtype=self.dtype)(x)  # N,H,W,336

        stride = x.shape[1] // 7 if x.shape[1] > 7 else 1
        adap_kernel = x.shape[1] - (7 - 1) * stride if x.shape[1] > 7 else x.shape[1]
        x = nn.avg_pool(x, (adap_kernel, adap_kernel), strides=(stride, stride))
        x = x.reshape((x.shape[0], -1))
        logits = nn.Sequential(
            [
                nn.Dropout(self.dropout_rate, deterministic=not training)
                if self.use_dropout
                else lambda x: x,
                nn.Dense(384, dtype=self.dtype),
                nn.Dense(192, dtype=self.dtype),
                nn.Dense(self.num_classes, dtype=self.dtype),
            ]
        )(x)
        return logits


class TrainState(train_state.TrainState):
    batch_stats: Any


def default_config():
    config = ml_collections.ConfigDict()

    config.model_name = "inception"
    config.dataset_type = "normal_labels"
    config.dtype = jnp.float32

    # dataset
    config.dataset = ml_collections.ConfigDict()
    config.dataset.name = "cifar10"
    config.dataset.num_classes = 10
    config.dataset.image_size = 32
    config.dataset.num_train = 50000

    # optim
    config.optim = ml_collections.ConfigDict()
    config.optim.name = "sgd"
    config.optim.learning_rate = 0.04
    config.optim.momentum = 0.9

    # train
    config.batch_size = 256
    config.num_epochs = 20
    config.num_steps_per_epoch = (
        len(range(config.dataset.num_train)) // config.batch_size
    )
    config.num_steps = config.num_epochs * config.num_steps_per_epoch

    config.eval_frequency = 5
    config.test_frequency = 5
    return config


def create_train_state(
    config: ml_collections.ConfigDict = None,
    rng: jax.random.PRNGKey = None,
) -> Dict[str, Any]:
    """Creates initial `TrainState`."""
    if rng is None:
        rng = jax.random.PRNGKey(0)

    rng, dropout_rng = jax.random.split(rng)
    rngs = {"params": rng, "dropout": dropout_rng}

    if config is None:
        config = default_config()

    cnn = InceptionSmall(num_classes=config.dataset.num_classes, dtype=config.dtype)

    variables = cnn.init(
        rngs,
        jnp.ones(
            [1, config.image_size, config.dataset.image_size, 3], dtype=config.dtype
        ),
    )  #  Jax/Flax/TF use NHWC.

    tx = optax.chain(
        optax.sgd(config.optim.learning_rate, momentum=config.optim.momentum),
        optax.clip_by_global_norm(1.0),
    )

    return TrainState.create(
        apply_fn=cnn.apply,
        params=variables["params"],
        batch_stats=variables["batch_stats"],
        tx=tx,
    )


@jax.jit
def train_step(
    state: train_state.TrainState, x: jnp.ndarray, y: jnp.ndarray, rngs
) -> Tuple[train_state.TrainState, Dict[str, Any]]:
    """Train for a single step.

    :param state: Current state
    :param x: Input
    :param y: Labels
    :param rngs: Random number generators for dropout

    :returns: New state, metrics dictionary
    """

    def loss_fn(params):
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            x=x,
            training=True,
            rngs=rngs,
            mutable=["batch_stats"],
        )
        one_hot = jax.nn.one_hot(y, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, (logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    metrics = {
        "loss": loss,
        "accuracy": jnp.mean(jnp.argmax(logits, -1) == y),
        "logits": logits,
    }
    return state, metrics


if __name__ == "__main__":
    import os
    import time

    import jax.numpy as jnp
    import optax
    from flax.training import train_state

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    rng = jax.random.PRNGKey(0)
    rng, dropout_rng = jax.random.split(rng)
    rngs = {"params": rng, "dropout": dropout_rng}

    config = default_config()
    config.dtype = jnp.float16

    s_time = time.time()
    state = create_train_state(config=config, rng=rng)
    print(f"init takes={time.time()-s_time:.3f}s")
    x = jnp.ones((64, 224, 224, 3))
    y = jnp.ones((64, 1), dtype=jnp.int32)

    s_time = time.time()

    for i in range(20):
        rng, epoch_rng = jax.random.split(rng)
        rngs = {"dropout": dropout_rng}
        s_time = time.time()
        state, metrics = train_step(state, x, y, rngs)
        print(
            f"iter={i:03d}, loss={metrics['loss']:.3f}, acc@1={metrics['accuracy']:.3f}, time={time.time() - s_time:.3f}s",
            metrics["logits"].shape,
            metrics["logits"].dtype,
        )


def inception(cifar=False, **kwargs):
    if cifar:
        return InceptionSmall(**kwargs)

    raise NotImplementedError
