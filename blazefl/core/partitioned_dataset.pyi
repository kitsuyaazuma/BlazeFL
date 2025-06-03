from abc import ABC, abstractmethod
from torch.utils.data import DataLoader, Dataset

class PartitionedDataset(ABC):
    @abstractmethod
    def get_dataset(self, type_: str, cid: int | None) -> Dataset: ...
    @abstractmethod
    def get_dataloader(self, type_: str, cid: int | None, batch_size: int | None) -> DataLoader: ...
