from abc import ABC, abstractmethod

from torch.utils.data import DataLoader, Dataset


class PartitionedDataset(ABC):
    """
    Abstract base class for partitioned datasets in federated learning.

    This class defines the interface for managing datasets that are partitioned
    across multiple clients.

    Raises:
        NotImplementedError: If the methods are not implemented in a subclass.
    """

    @abstractmethod
    def get_dataset(self, type_: str, cid: int | None) -> Dataset:
        """
        Retrieve a dataset for a specific type and client ID.

        Args:
            type_ (str): The type of the dataset.
            cid (int | None): The client ID.

        Returns:
            Dataset: The dataset.
        """
        ...

    @abstractmethod
    def get_dataloader(
        self, type_: str, cid: int | None, batch_size: int | None
    ) -> DataLoader:
        """
        Retrieve a DataLoader for a specific type, client ID, and batch size.

        Args:
            type_ (str): The type of the dataset.
            cid (int | None): The client ID.
            batch_size (int | None): The batch size.

        Returns:
            DataLoader: The DataLoader.
        """
        ...
