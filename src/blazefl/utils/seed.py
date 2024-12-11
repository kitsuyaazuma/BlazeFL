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


@dataclass
class RandomState:
    random: tuple
    environ: str
    numpy: dict
    torch: int
    cuda: CUDARandomState | None

    @classmethod
    def get_random_state(cls, device: str) -> "RandomState":
        if device.startswith("cuda"):
            return cls(
                random.getstate(),
                os.environ["PYTHONHASHSEED"],
                np.random.get_state(),
                torch.initial_seed(),
                CUDARandomState(
                    torch.cuda.initial_seed(),
                    torch.backends.cudnn.deterministic,
                    torch.backends.cudnn.benchmark,
                ),
            )
        return cls(
            random.getstate(),
            os.environ["PYTHONHASHSEED"],
            np.random.get_state(),
            torch.initial_seed(),
            None,
        )

    @staticmethod
    def set_random_state(random_state: "RandomState") -> None:
        random.setstate(random_state.random)
        os.environ["PYTHONHASHSEED"] = random_state.environ
        np.random.set_state(random_state.numpy)
        if random_state.cuda is not None:
            torch.manual_seed(random_state.torch)
            torch.cuda.manual_seed(random_state.cuda.manual_seed)
            torch.backends.cudnn.deterministic = random_state.cuda.cudnn_deterministic
            torch.backends.cudnn.benchmark = random_state.cuda.cudnn_benchmark
