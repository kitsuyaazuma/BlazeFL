from collections.abc import Callable

from torch.utils.data import Dataset


class FilteredDataset(Dataset):
    """
    A dataset wrapper that filters and transforms a subset of the original dataset.

    This class allows selecting specific data points by their indices and
    applying optional transformations to the data and targets.

    Attributes:
        data (list): The filtered subset of the original dataset.
        targets (list | None): The filtered subset of targets, if provided.
        transform (Callable | None): A function to apply transformations to the data.
        target_transform (Callable | None): A function to apply
        transformations to the targets.
    """

    def __init__(
        self,
        indices: list[int],
        original_data: list,
        original_targets: list | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """
        Initialize the FilteredDataset.

        Args:
            indices (list[int]): Indices of the data points to include in the dataset.
            original_data (list): The original dataset.
            original_targets (list | None): The original targets, if available.
            transform (Callable | None): Transformation function for the data.
            target_transform (Callable | None): Transformation function for the targets.
        """
        self.data = [original_data[i] for i in indices]
        if original_targets is not None:
            assert len(original_data) == len(original_targets)
            self.targets = [original_targets[i] for i in indices]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        """
        Return the length of the filtered dataset.

        Returns:
            int: The number of data points in the dataset.
        """
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        """
        Retrieve a data item (and optionally its target) at a specific index.

        Args:
            index (int): The index of the data point to retrieve.

        Returns:
            tuple: A tuple containing the transformed data item and its target
            (if available).
        """
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        if hasattr(self, "targets"):
            target = self.targets[index]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

        return img
