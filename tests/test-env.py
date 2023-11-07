"""
Check virtual environment and CUDA availability

- Checks if torch is available
- Checks if jax is available
- Checks if torch + CUDA is available
- Checks if jax + GPU is available
"""


def test_torch():
    try:
        import torch
    except ImportError:
        print("Torch is not available")


def test_jax():
    try:
        import jax
    except ImportError:
        print("Jax is not available")


def test_flax():
    try:
        import flax
    except ImportError:
        print("Flax is not available")


def test_torch_cuda():
    import torch

    assert torch.cuda.is_available() == True, "Torch + CUDA is not available"


def test_jax_gpu():
    import jax

    assert jax.devices("gpu") == True, "Jax + GPU is not available"
