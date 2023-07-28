from typing import Any, Dict

from fastprogress.fastprogress import master_bar


def init_metrics() -> Dict[str, Any]:
    return {
        "train_loss": [],
        "train_accuracy": [],
        "test_loss": [],
        "test_accuracy": [],
        "test_top_5_accuracy": [],
        "train_sample_losses": [],
        "test_sample_losses": [],
        "train_epoch_time": [],
        "test_epoch_time": [],
    }


def update_metrics(
    train: bool,
    metrics: Dict[str, Any],
    loss,
    acc,
    loss_per_sample,
    epoch_time: float,
    top_5_acc=None,
    trainloader_size=None,
):
    if train:
        metrics["train_loss"].append(loss / trainloader_size)
        metrics["train_accuracy"].append(acc / trainloader_size)
        metrics["train_sample_losses"].append(loss_per_sample)
        metrics["train_epoch_time"].append(epoch_time)
    else:
        metrics["test_loss"].append(loss)
        metrics["test_accuracy"].append(acc)
        metrics["test_top_5_accuracy"].append(top_5_acc)
        metrics["test_sample_losses"].append(loss_per_sample)
        metrics["test_epoch_time"].append(epoch_time)


def init_bar(epochs, metrics) -> master_bar:
    mbar = master_bar(range(epochs))
    mbar.names = ["epoch"]
    for name in metrics:
        # filter per-sample metrics
        if "sample" in name:
            continue
        mbar.names.append(name)
    mbar.write(mbar.names, table=True)
    return mbar


def update_mbar(epoch, metrics, mbar):
    update = [epoch]
    update += [f"{metrics[name][-1]:.3f}" for name in mbar.names[1:]]
    mbar.write(update, table=True)
