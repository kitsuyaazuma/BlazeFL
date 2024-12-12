import random

import numpy as np
import pytest
import torch

from src.blazefl.utils import RandomState, seed_everything


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_seed_everything(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    seed = 42
    seed_everything(seed, device)

    py_rand_val = random.random()
    np_rand_val = np.random.rand()
    torch_rand_val = torch.rand(1, device=device)

    seed_everything(seed, device)
    assert py_rand_val == random.random()
    assert np.allclose(np_rand_val, np.random.rand())
    assert torch.allclose(torch_rand_val, torch.rand(1, device=device))


def test_random_state_cpu() -> None:
    device = "cpu"
    seed = 123
    seed_everything(seed, device)

    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(1, device=device)

    state = RandomState.get_random_state(device)

    py_rand_val_before = random.random()
    np_rand_val_before = np.random.rand()
    torch_rand_val_before = torch.rand(1, device=device)

    RandomState.set_random_state(state)

    assert py_rand_val_before == random.random()
    assert np.allclose(np_rand_val_before, np.random.rand())
    assert torch.allclose(torch_rand_val_before, torch.rand(1, device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_random_state_cuda() -> None:
    device = "cuda"
    seed = 123
    seed_everything(seed, device)

    _ = random.random()
    _ = np.random.rand()
    _ = torch.rand(1, device=device)

    state = RandomState.get_random_state(device)

    py_rand_val_before = random.random()
    np_rand_val_before = np.random.rand()
    torch_rand_val_before = torch.rand(1, device=device)

    RandomState.set_random_state(state)

    assert py_rand_val_before == random.random()
    assert np.allclose(np_rand_val_before, np.random.rand())
    assert torch.allclose(torch_rand_val_before, torch.rand(1, device=device))
