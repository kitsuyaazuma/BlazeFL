from pathlib import Path
from typing import Any
import torch


def set_shared_memory(
    parameters: dict[str, Any],
) -> None:
    for v in parameters.values():
        if isinstance(v, torch.Tensor):
            v.cpu().share_memory_()


def set_shared_disk(
    parameters: dict[str, Any],
    path: Path,
) -> None:
    torch.save(parameters, path)


def get_shared_disk(path: Path) -> dict[str, Any]:
    parameters = torch.load(path, weights_only=False)
    return parameters
