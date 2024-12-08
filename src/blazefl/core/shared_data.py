from pathlib import Path
from typing import Generic, TypeVar

import torch
from pydantic import BaseModel

SharedMemoryDataType = TypeVar("SharedMemoryDataType", bound=BaseModel)
DiskDataType = TypeVar("DiskDataType", bound=BaseModel)


class SharedData(Generic[SharedMemoryDataType, DiskDataType]):
    def __init__(
        self,
        shared_memory_data: SharedMemoryDataType,
        disk_data: DiskDataType,
        disk_path: Path,
    ) -> None:
        self.shared_memory_data = shared_memory_data
        self.disk_data = disk_data
        self.disk_data_type = type(self.disk_data)
        self.disk_path = disk_path

    def share(self) -> "SharedData[SharedMemoryDataType, DiskDataType]":
        for field_key in self.shared_memory_data.model_fields:
            field_value = getattr(self.shared_memory_data, field_key)
            if isinstance(field_value, torch.Tensor):
                field_value.share_memory_()
        if self.disk_data:
            self.disk_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.disk_data, self.disk_path)
            del self.disk_data
        return self

    def get_shared_memory_data(self) -> SharedMemoryDataType:
        return self.shared_memory_data

    def get_disk_data(self) -> DiskDataType:
        assert self.disk_path.exists()
        loaded_data = torch.load(self.disk_path, weights_only=False)
        assert isinstance(loaded_data, BaseModel)
        return self.disk_data_type(**loaded_data.model_dump())
