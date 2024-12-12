import os
import random
from dataclasses import dataclass

import numpy as np
import torch


def seed_everything(seed: int, device: str) -> None:
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
    manual_seed: int
    cudnn_deterministic: bool
    cudnn_benchmark: bool
    cuda_rng_state: torch.Tensor


@dataclass
class RandomState:
    random: tuple
    environ: str
    numpy: dict
    torch_seed: int
    torch_rng_state: torch.Tensor
    cuda: CUDARandomState | None

    @classmethod
    def get_random_state(cls, device: str) -> "RandomState":
        if device.startswith("cuda"):
            return cls(
                random.getstate(),
                os.environ["PYTHONHASHSEED"],
                np.random.get_state(),
                torch.initial_seed(),
                torch.cuda.get_rng_state(),
                CUDARandomState(
                    torch.cuda.initial_seed(),
                    torch.backends.cudnn.deterministic,
                    torch.backends.cudnn.benchmark,
                    torch.cuda.get_rng_state(),
                ),
            )
        return cls(
            random.getstate(),
            os.environ["PYTHONHASHSEED"],
            np.random.get_state(),
            torch.initial_seed(),
            torch.get_rng_state(),
            None,
        )

    @staticmethod
    def set_random_state(random_state: "RandomState") -> None:
        random.setstate(random_state.random)
        os.environ["PYTHONHASHSEED"] = random_state.environ
        np.random.set_state(random_state.numpy)
        torch.manual_seed(random_state.torch_seed)
        if random_state.cuda is not None:
            torch.cuda.manual_seed(random_state.cuda.manual_seed)
            torch.backends.cudnn.deterministic = random_state.cuda.cudnn_deterministic
            torch.backends.cudnn.benchmark = random_state.cuda.cudnn_benchmark
            torch.cuda.set_rng_state(random_state.cuda.cuda_rng_state)
        else:
            torch.set_rng_state(random_state.torch_rng_state)
