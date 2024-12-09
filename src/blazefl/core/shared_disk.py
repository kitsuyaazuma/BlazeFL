from pathlib import Path
from typing import Generic, TypeVar

import torch

T = TypeVar("T")


class SharedDisk(Generic[T]):
    def __init__(self, data: T, path: Path, share: bool = True) -> None:
        self.data = data
        self.data_type = type(self.data)
        self.path = path
        if share:
            self.share()

    def share(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.data, self.path)
        del self.data

    def get_data(self) -> T:
        assert self.path.exists()
        loaded_data = torch.load(self.path, weights_only=False)
        assert isinstance(loaded_data, self.data_type)
        return loaded_data
