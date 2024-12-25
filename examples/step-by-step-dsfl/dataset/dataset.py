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


class DSFLPartitionedDataset(PartitionedDataset):
    def __init__(
        self,
        root: Path,
        path: Path,
        num_clients: int,
        seed: int,
        partition: str,
        open_size: int,
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
        self.open_size = open_size
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
        openset = torchvision.datasets.CIFAR100(
            root=self.root,
            train=True,
            download=True,
        )
        testset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
        )
        for type_ in ["train", "val", "open", "test"]:
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

        open_indices, _ = train_test_split(
            range(len(openset)),
            test_size=1 - self.open_size / len(openset),
        )
        torch.save(
            FilteredDataset(
                open_indices,
                openset.data,
                original_targets=None,
                transform=self.transform,
                target_transform=self.target_transform,
            ),
            self.path.joinpath("open", "open.pkl"),
        )

        torch.save(
            FilteredDataset(
                list(range(len(testset))),
                testset.data,
                testset.targets,
                transform=self.transform,
                target_transform=self.target_transform,
            ),
            self.path.joinpath("test", "test.pkl"),
        )

    def get_dataset(self, type_: str, cid: int | None) -> Dataset:
        match type_:
            case "train" | "val":
                dataset = torch.load(
                    self.path.joinpath(type_, f"{cid}.pkl"),
                    weights_only=False,
                )
            case "open":
                dataset = torch.load(
                    self.path.joinpath(type_, "open.pkl"),
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
