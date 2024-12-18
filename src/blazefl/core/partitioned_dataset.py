from abc import ABC, abstractmethod

from torch.utils.data import DataLoader, Dataset


class PartitionedDataset(ABC):
    """
    Abstract class for partitioned datasets.
    """

    @abstractmethod
    def get_dataset(self, type_: str, cid: int | None) -> Dataset:
        """
        Get a dataset for a specific type and client ID.

        Args:
            type_ (str): The type of the dataset.
            cid (int | None): The client ID.
        """
        ...

    @abstractmethod
    def get_dataloader(
        self, type_: str, cid: int | None, batch_size: int | None
    ) -> DataLoader: ...
