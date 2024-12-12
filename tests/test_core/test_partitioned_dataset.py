from torch.utils.data import DataLoader, Dataset

from src.blazefl.core import PartitionedDataset


def test_partitioned_dataset_subclass() -> None:
    class DummyDataset(Dataset):
        def __len__(self):
            return 10

        def __getitem__(self, index):
            return index

    class DummyPartitionedDataset(PartitionedDataset):
        def get_dataset(self, type_: str, cid: int | None) -> DummyDataset:
            _ = (type_, cid)
            return DummyDataset()

        def get_dataloader(
            self, type_: str, cid: int | None, batch_size: int | None
        ) -> DataLoader:
            dataset = self.get_dataset(type_, cid)
            if batch_size is None:
                batch_size = len(dataset)
            return DataLoader(dataset, batch_size=batch_size)

    partitioned_dataset = DummyPartitionedDataset()

    dataset = partitioned_dataset.get_dataset("train", 0)
    assert len(dataset) == 10
    assert dataset[0] == 0

    dataloader = partitioned_dataset.get_dataloader("test", None, batch_size=2)
    batch = next(iter(dataloader))
    assert len(batch) == 2
    assert batch[0] == 0
    assert batch[1] == 1
