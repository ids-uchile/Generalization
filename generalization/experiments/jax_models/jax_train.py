import time
from typing import Any, Dict, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
from fastprogress.fastprogress import progress_bar
from jaxtyping import Array, Float, Int, PyTree

from .utils import init_bar, init_metrics, update_mbar, update_metrics

key = jax.random.PRNGKey(42)


@eqx.filter_jit
def loss(
    net: eqx.Module, batch: Float[Array, "batch 3 28 28"], labels: Int[Array, " batch"]
) -> Float[Array, ""]:
    """Compute the loss of the network on a batch of data."""
    batch_size = batch.shape[0]
    batched_keys = jax.random.split(key, num=batch_size)

    logits = jax.vmap(net, in_axes=(0, 0))(batch, batched_keys)

    one_hot_labels = jax.nn.one_hot(labels, 10)

    # optax also provides a number of common loss functions.
    loss_per_sample = optax.softmax_cross_entropy(logits, one_hot_labels)

    aux = {
        "loss": jnp.mean(loss_per_sample),
        "accuracy": jnp.mean(jnp.argmax(logits, axis=-1) == labels),
        "top_5_accuracy": jnp.mean(
            jnp.sum(jnp.argsort(logits, axis=-1)[:, -5:] == labels[:, None], axis=-1)
        ),
        "loss_per_sample": loss_per_sample,
    }
    return jnp.mean(loss_per_sample), aux


def evaluate(model: eqx.Module, testloader: torch.utils.data.DataLoader):
    """This function evaluates the model on the test dataset,
    computing both the average loss and the average accuracy.
    """
    avg_loss = 0
    avg_acc = 0
    avg_top_5_acc = 0
    sample_losses = []
    for x, y in testloader:
        x = x.numpy()
        y = y.numpy()
        # Note that all the JAX operations happen inside `loss` and `compute_accuracy`,
        # and both have JIT wrappers, so this is fast.
        out_loss, aux = loss(model, x, y)
        avg_loss += out_loss
        avg_acc += aux["accuracy"]
        avg_top_5_acc += aux["top_5_accuracy"]

        sample_losses.append(aux["loss_per_sample"])

    return (
        avg_loss / len(testloader),
        avg_acc / len(testloader),
        avg_top_5_acc / len(testloader),
        sample_losses,
    )


def train(
    model: eqx.Module,
    trainloader: torch.utils.data.DataLoader,
    testloader: torch.utils.data.DataLoader,
    optim: optax.GradientTransformation,
    epochs: int,
    print_every: int,
) -> Union[eqx.Module, Dict[str, Any]]:
    # Just like earlier: It only makes sense to train the arrays in our model,
    # so filter out everything else.
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    # Always wrap everything -- computing gradients, running the optimiser, updating
    # the model -- into a single JIT region. This ensures things run as fast as
    # possible.
    @eqx.filter_jit
    def make_step(
        model: eqx.Module,
        opt_state: PyTree,
        x: Float[Array, "batch 3 28 28"],
        y: Int[Array, " batch"],
    ):
        (loss_value, aux), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
            model, x, y
        )
        updates, opt_state = optim.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss_value, aux

    metrics = init_metrics()
    mbar = init_bar(epochs=epochs, metrics=metrics)

    start_time = time.time()
    for epoch in mbar:
        start_epoch_time = time.time()
        running_loss = 0.0
        runnning_acc = 0.0
        running_sample_losses = []

        for _, (x, y) in enumerate(progress_bar(trainloader, parent=mbar)):
            # PyTorch dataloaders give PyTorch tensors by default,
            # so convert them to NumPy arrays.
            x = x.numpy()
            y = y.numpy()
            model, opt_state, train_loss, aux = make_step(model, opt_state, x, y)

            running_loss += train_loss.item()
            runnning_acc += aux["accuracy"].item()
            running_sample_losses.append(aux["loss_per_sample"])
        end_epoch_time = time.time()

        update_metrics(
            train=True,
            metrics=metrics,
            loss=running_loss,
            acc=runnning_acc,
            loss_per_sample=running_sample_losses,
            epoch_time=end_epoch_time - start_epoch_time,
            trainloader_size=len(trainloader),
        )

        if (epoch % print_every) == 0 or (epochs == epoch):
            start_test_time = time.time()
            test_loss, test_accuracy, top_5_acc, test_sample_losses = evaluate(
                model, testloader
            )
            end_test_time = time.time()
            update_metrics(
                train=False,
                metrics=metrics,
                loss=test_loss,
                acc=test_accuracy,
                loss_per_sample=test_sample_losses,
                top_5_acc=top_5_acc,
                epoch_time=end_test_time - start_test_time,
            )

        update_mbar(epoch, metrics, mbar)

    print(
        f"Total time: {time.time() - start_time:.3f} s, {(time.time() - start_time)/ 60:.3f} min"
    )
    metrics["total_time"] = time.time() - start_time
    return model, metrics
