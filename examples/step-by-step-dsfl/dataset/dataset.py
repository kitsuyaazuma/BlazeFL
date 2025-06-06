from collections.abc import Sized
from pathlib import Path

import numpy as np
import torch
import torchvision
from blazefl.core import PartitionedDataset
from blazefl.utils import FilteredDataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataset.functional import (
    balance_split,
    client_inner_dirichlet_partition_faster,
)


class DSFLPartitionedDataset(PartitionedDataset):
    def __init__(
        self,
        root: Path,
        path: Path,
        num_clients: int,
        seed: int,
        partition: str,
        open_size: int,
        dir_alpha: float,
    ) -> None:
        self.root = root
        self.path = path
        self.num_clients = num_clients
        self.seed = seed
        self.partition = partition
        self.dir_alpha = dir_alpha
        self.open_size = open_size

        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self._preprocess()

    def _preprocess(self):
        self.root.mkdir(parents=True, exist_ok=True)
        train_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
        )
        open_dataset = torchvision.datasets.CIFAR100(
            root=self.root,
            train=True,
            download=True,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
        )
        for type_ in ["train", "open", "test"]:
            self.path.joinpath(type_).mkdir(parents=True)

        match self.partition:
            case "client_inner_dirichlet":
                client_dict, class_priors = client_inner_dirichlet_partition_faster(
                    targets=train_dataset.targets,
                    num_clients=self.num_clients,
                    num_classes=10,
                    dir_alpha=self.dir_alpha,
                    client_sample_nums=balance_split(
                        num_clients=self.num_clients,
                        num_samples=len(train_dataset.targets),
                    ),
                    verbose=False,
                )
                test_client_dict, _ = client_inner_dirichlet_partition_faster(
                    targets=test_dataset.targets,
                    num_clients=self.num_clients,
                    num_classes=10,
                    dir_alpha=self.dir_alpha,
                    client_sample_nums=balance_split(
                        num_clients=self.num_clients,
                        num_samples=len(test_dataset.targets),
                    ),
                    class_priors=class_priors,
                    verbose=False,
                )
            case _:
                raise ValueError(f"Invalid partition: {self.partition}")

        for cid, indices in client_dict.items():
            client_train_dataset = FilteredDataset(
                indices.tolist(),
                train_dataset.data,
                train_dataset.targets,
                transform=self.train_transform,
            )
            torch.save(client_train_dataset, self.path.joinpath("train", f"{cid}.pkl"))

        for cid, indices in test_client_dict.items():
            client_test_dataset = FilteredDataset(
                indices.tolist(),
                test_dataset.data,
                test_dataset.targets,
                transform=self.test_transform,
            )
            torch.save(client_test_dataset, self.path.joinpath("test", f"{cid}.pkl"))

        open_indices = np.random.choice(
            len(open_dataset),
            size=self.open_size,
        )
        torch.save(
            FilteredDataset(
                open_indices.tolist(),
                open_dataset.data,
                original_targets=None,
                transform=self.train_transform,
            ),
            self.path.joinpath("open.pkl"),
        )

        torch.save(
            FilteredDataset(
                list(range(len(test_dataset))),
                test_dataset.data,
                test_dataset.targets,
                transform=self.test_transform,
            ),
            self.path.joinpath("test", "default.pkl"),
        )

    def get_dataset(self, type_: str, cid: int | None) -> Dataset:
        match type_:
            case "train":
                dataset = torch.load(
                    self.path.joinpath(type_, f"{cid}.pkl"),
                    weights_only=False,
                )
            case "open":
                dataset = torch.load(
                    self.path.joinpath(f"{type_}.pkl"),
                    weights_only=False,
                )
            case "test":
                if cid is not None:
                    dataset = torch.load(
                        self.path.joinpath(type_, f"{cid}.pkl"),
                        weights_only=False,
                    )
                else:
                    dataset = torch.load(
                        self.path.joinpath(type_, "default.pkl"), weights_only=False
                    )
            case _:
                raise ValueError(f"Invalid dataset type: {type_}")
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
