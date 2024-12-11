from collections.abc import Callable, Sized
from pathlib import Path

import torch
import torchvision
from fedlab.utils.dataset.functional import (
    balance_split,
    client_inner_dirichlet_partition,
    shards_partition,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from blazefl.core import PartitionedDataset
from blazefl.utils import FilteredDataset


class PartitionedCIFAR10(PartitionedDataset):
    def __init__(
        self,
        root: Path,
        path: Path,
        num_clients: int,
        seed: int,
        partition: str,
        num_shards: int | None = None,
        dir_alpha: float | None = None,
        val_ratio: float = 0.2,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        self.root = root
        self.path = path
        self.num_clients = num_clients
        self.seed = seed
        self.partition = partition
        self.num_shards = num_shards
        self.dir_alpha = dir_alpha
        self.val_ratio = val_ratio
        self.transform = transform
        self.target_transform = target_transform

        self._preprocess()

    def _preprocess(self):
        self.root.mkdir(parents=True, exist_ok=True)
        trainset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
        )
        testset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
        )
        for type_ in ["train", "val", "test"]:
            self.path.joinpath(type_).mkdir(parents=True)

        match self.partition:
            case "client_inner_dirichlet":
                assert self.dir_alpha is not None
                client_dict = client_inner_dirichlet_partition(
                    targets=trainset.targets,
                    num_clients=self.num_clients,
                    num_classes=10,
                    dir_alpha=self.dir_alpha,
                    client_sample_nums=balance_split(
                        len(trainset.targets), self.num_clients
                    ),
                )
            case "shards":
                assert self.num_shards is not None
                client_dict = shards_partition(
                    targets=trainset.targets,
                    num_clients=self.num_clients,
                    num_shards=self.num_shards,
                )
            case _:
                raise ValueError("Invalid partition")

        for cid, indices in client_dict.items():
            train_indices, val_indices = train_test_split(
                indices,
                test_size=self.val_ratio,
            )
            client_trainset = FilteredDataset(
                train_indices,
                trainset.data,
                trainset.targets,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            torch.save(client_trainset, self.path.joinpath("train", f"{cid}.pkl"))
            client_valset = FilteredDataset(
                val_indices,
                trainset.data,
                trainset.targets,
                transform=self.transform,
                target_transform=self.target_transform,
            )
            torch.save(client_valset, self.path.joinpath("val", f"{cid}.pkl"))

        torch.save(
            testset,
            self.path.joinpath("test", "test.pkl"),
        )

    def get_dataset(self, type_: str, cid: int | None) -> Dataset:
        match type_:
            case "train" | "val":
                dataset = torch.load(
                    self.path.joinpath(type_, f"{cid}.pkl"),
                    weights_only=False,
                )
            case "test":
                dataset = torch.load(
                    self.path.joinpath(type_, "test.pkl"), weights_only=False
                )
            case _:
                raise ValueError("Invalid type_")
        assert isinstance(dataset, Dataset)
        return dataset

    def get_dataloader(
        self, type_: str, cid: int | None, batch_size: int | None = None
    ) -> DataLoader:
        dataset = self.get_dataset(type_, cid)
        assert isinstance(dataset, Sized)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader
