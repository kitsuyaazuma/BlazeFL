from enum import Enum
from pathlib import Path
from typing import Any, TypeVar, Type
import torch


class ShareMethod(Enum):
    SHARED_MEMORY = "shared_memory"
    DISK = "disk"


T = TypeVar("T", bound="SharedData")


class SharedData(object):
    def __init__(self, disk_path: Path | None = None) -> None:
        self.shared_memory_data: dict[str, Any] = {}
        self.disk_data: dict[str, Any] = {}
        self.disk_path = disk_path

    def set(
        self, key: str, value: Any, method: ShareMethod = ShareMethod.SHARED_MEMORY
    ) -> "SharedData":
        match method:
            case ShareMethod.SHARED_MEMORY:
                self.shared_memory_data[key] = value
            case ShareMethod.DISK:
                self.disk_data[key] = value
        return self

    @classmethod
    def copy(cls: Type[T], instance: T) -> T:
        copied_instance = cls(instance.disk_path)
        copied_instance.shared_memory_data = instance.shared_memory_data.copy()
        copied_instance.disk_data = instance.disk_data.copy()
        return copied_instance

    def share(self) -> None:
        for v in self.shared_memory_data.values():
            if isinstance(v, torch.Tensor):
                v.cpu().share_memory_()
        if len(self.disk_data) > 0:
            assert self.disk_path is not None, "Disk path is not provided."
            self.disk_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.disk_data, self.disk_path)
            self.disk_data = {}

    def get(self, key: str, method: ShareMethod = ShareMethod.SHARED_MEMORY) -> Any:
        match method:
            case ShareMethod.SHARED_MEMORY:
                return self.shared_memory_data[key]
            case ShareMethod.DISK:
                assert self.disk_path is not None and self.disk_path.exists()
                return torch.load(self.disk_path, weights_only=False)[key]

    def get_all(self, method: ShareMethod | None = None) -> dict[str, Any]:
        match method:
            case ShareMethod.SHARED_MEMORY:
                return self.shared_memory_data
            case ShareMethod.DISK:
                assert self.disk_path is not None and self.disk_path.exists()
                return torch.load(self.disk_path, weights_only=False)
            case None:
                data = {}
                data.update(self.shared_memory_data)
                if self.disk_path is not None and self.disk_path.exists():
                    data.update(torch.load(self.disk_path, weights_only=False))
                return data

    def clean(self) -> None:
        self.shared_memory_data = {}
        self.disk_data = {}
        if self.disk_path is not None and self.disk_path.exists():
            self.disk_path.unlink()
