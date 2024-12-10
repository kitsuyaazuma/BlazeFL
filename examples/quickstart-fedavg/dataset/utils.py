from collections.abc import Callable

from torch.utils.data import Dataset


class FilteredDataset(Dataset):
    def __init__(
        self,
        indices: list[int],
        original_data: list,
        original_targets: list | None = None,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        self.data = [original_data[i] for i in indices]
        if original_targets is not None:
            assert len(original_data) == len(original_targets)
            self.targets = [original_targets[i] for i in indices]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple:
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        if hasattr(self, "targets"):
            target = self.targets[index]
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

        return img
