import os
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch


def seed_everything(seed: int, device: str) -> None:
    """
    Seed random number generators for reproducibility.

    This function sets seeds for Python's random module, NumPy, and PyTorch
    to ensure deterministic behavior in experiments.

    Args:
        seed (int): The seed value to set.
        device (str): The device type ('cpu' or 'cuda').

    Returns:
        None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class CUDARandomState:
    """
    A dataclass representing the random state for CUDA.

    Attributes:
        manual_seed (int): The manual seed for CUDA.
        cudnn_deterministic (bool): The deterministic setting for cuDNN.
        cudnn_benchmark (bool): The benchmark setting for cuDNN.
        cuda_rng_state (torch.Tensor): The RNG state for CUDA.
    """

    manual_seed: int
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_rng_state: torch.Tensor


@dataclass
class RandomState:
    """
    A dataclass representing the random state for Python, NumPy, and PyTorch.

    Attributes:
        _random (tuple): The state of Python's random module.
        _environ (str): The PYTHONHASHSEED environment variable.
        _numpy (tuple): The state of NumPy's RNG.
        _torch_seed (int): The initial seed for PyTorch.
        _torch_rng_state (torch.Tensor): The RNG state for PyTorch.
        _cuda (CUDARandomState | None): The CUDA-specific random state, if available.
    """

    _random: tuple[Any, ...]
    _environ: str
    _numpy: tuple[str, npt.NDArray[np.uint32], int, int, float]
    _torch_seed: int
    _torch_rng_state: torch.Tensor
    _cuda: CUDARandomState | None

    @classmethod
    def get_random_state(cls, device: str) -> "RandomState":
        """
        Capture the current random state.

        Args:
            device (str): The device type ('cpu' or 'cuda').

        Returns:
            RandomState: The captured random state.
        """
        _random = random.getstate()
        _environ = os.environ["PYTHONHASHSEED"]
        _numpy = np.random.get_state(legacy=True)
        assert isinstance(_numpy, tuple)
        _torch_seed = torch.initial_seed()
        _torch_rng_state = torch.get_rng_state()

        random_state = cls(
            _random, _environ, _numpy, _torch_seed, _torch_rng_state, None
        )
        if device.startswith("cuda"):
            random_state._torch_rng_state = torch.cuda.get_rng_state()
            random_state._cuda = CUDARandomState(
                torch.cuda.initial_seed(),
                torch.backends.cudnn.deterministic,
                torch.backends.cudnn.benchmark,
                torch.cuda.get_rng_state(),
            )
        return random_state

    @staticmethod
    def set_random_state(random_state: "RandomState") -> None:
        """
        Restore the random state from a RandomState object.

        Args:
            random_state (RandomState): The random state to restore.

        Returns:
            None
        """
        random.setstate(random_state._random)
        os.environ["PYTHONHASHSEED"] = random_state._environ
        np.random.set_state(random_state._numpy)
        torch.manual_seed(random_state._torch_seed)
        if random_state._cuda is not None:
            torch.cuda.manual_seed(random_state._cuda.manual_seed)
            torch.backends.cudnn.deterministic = random_state._cuda.cudnn_deterministic
            torch.backends.cudnn.benchmark = random_state._cuda.cudnn_benchmark
            torch.cuda.set_rng_state(random_state._cuda.cuda_rng_state)
        else:
            torch.set_rng_state(random_state._torch_rng_state)
