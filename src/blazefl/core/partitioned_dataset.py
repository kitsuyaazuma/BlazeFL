from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from typing import Optional


class PartitionedDataset(ABC):
    @abstractmethod
    def preprocess(self) -> None: ...

    @abstractmethod
    def get_dataset(self, type_: str, cid: Optional[int]) -> Dataset: ...

    @abstractmethod
    def get_dataloader(
        self, type_: str, cid: Optional[int], batch_size: Optional[int]
    ) -> DataLoader: ...
