import random
from collections import defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from dataset import DSFLPartitionedDataset
from models import DSFLModelSelector
from torch.utils.data import DataLoader, Subset

from blazefl.core import (
    ParallelClientTrainer,
    ServerHandler,
)
from blazefl.utils import (
    FilteredDataset,
    RandomState,
    seed_everything,
)


@dataclass
class DSFLUplinkPackage:
    soft_labels: torch.Tensor
    indices: torch.Tensor
    metadata: dict


@dataclass
class DSFLDownlinkPackage:
    soft_labels: torch.Tensor | None
    indices: torch.Tensor | None
    next_indices: torch.Tensor


class DSFLServerHandler(ServerHandler[DSFLUplinkPackage, DSFLDownlinkPackage]):
    def __init__(
        self,
        model_selector: DSFLModelSelector,
        model_name: str,
        dataset: DSFLPartitionedDataset,
        global_round: int,
        num_clients: int,
        sample_ratio: float,
        device: str,
        kd_epochs: int,
        kd_batch_size: int,
        kd_lr: float,
        era_temperature: float,
        open_size_per_round: int,
    ) -> None:
        self.model = model_selector.select_model(model_name)
        self.dataset = dataset
        self.global_round = global_round
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.device = device
        self.kd_epochs = kd_epochs
        self.kd_batch_size = kd_batch_size
        self.kd_lr = kd_lr
        self.era_temperature = era_temperature
        self.open_size_per_round = open_size_per_round

        self.client_buffer_cache: list[DSFLUplinkPackage] = []
        self.global_soft_labels: torch.Tensor | None = None
        self.global_indices: torch.Tensor | None = None
        self.num_clients_per_round = int(self.num_clients * self.sample_ratio)
        self.round = 0

    def sample_clients(self) -> list[int]:
        sampled_clients = random.sample(
            range(self.num_clients), self.num_clients_per_round
        )

        return sorted(sampled_clients)

    def get_next_indices(self) -> torch.Tensor:
        shuffled_indices = torch.randperm(self.dataset.open_size)
        return shuffled_indices[: self.open_size_per_round]

    def if_stop(self) -> bool:
        return self.round >= self.global_round

    def load(self, payload: DSFLUplinkPackage) -> bool:
        self.client_buffer_cache.append(payload)

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1
            self.client_buffer_cache = []
            return True
        else:
            return False

    def global_update(self, buffer) -> None:
        soft_labels_list = [ele.soft_labels for ele in buffer]
        indices_list = [ele.indices for ele in buffer]
        self.metadata_list = [ele.metadata for ele in buffer]

        soft_labels_stack: defaultdict[int, list[torch.Tensor]] = defaultdict(
            list[torch.Tensor]
        )
        for soft_labels, indices in zip(soft_labels_list, indices_list, strict=True):
            for soft_label, index in zip(soft_labels, indices, strict=True):
                soft_labels_stack[int(index.item())].append(soft_label)

        global_soft_labels: list[torch.Tensor] = []
        global_indices: list[int] = []
        for indices, soft_labels in soft_labels_stack.items():
            global_indices.append(indices)
            mean_soft_labels = torch.mean(torch.stack(soft_labels), dim=0)
            # Entropy Reduction Aggregation (ERA)
            era_soft_labels = F.softmax(mean_soft_labels / self.era_temperature, dim=0)
            global_soft_labels.append(era_soft_labels)

        DSFLServerHandler.distill(
            self.model,
            self.dataset,
            global_soft_labels,
            global_indices,
            self.kd_epochs,
            self.kd_batch_size,
            self.kd_lr,
            self.device,
        )

        self.global_soft_labels = torch.stack(global_soft_labels)
        self.global_indices = torch.tensor(global_indices)

    @staticmethod
    def distill(
        model: torch.nn.Module,
        dataset: DSFLPartitionedDataset,
        global_soft_labels: list[torch.Tensor],
        global_indices: list[int],
        kd_epochs: int,
        kd_batch_size: int,
        kd_lr: float,
        device: str,
    ) -> None:
        model.to(device)
        model.train()
        openset = dataset.get_dataset(type_="open", cid=None)
        open_loader = DataLoader(
            Subset(openset, global_indices),
            batch_size=kd_batch_size,
        )
        global_soft_label_loader = DataLoader(
            FilteredDataset(
                indices=list(range(len(global_soft_labels))),
                original_data=global_soft_labels,
            ),
            batch_size=kd_batch_size,
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=kd_lr)
        for _ in range(kd_epochs):
            for data, soft_label in zip(
                open_loader, global_soft_label_loader, strict=True
            ):
                data = data.to(device)
                soft_label = soft_label.to(device).squeeze(1)

                output = model(data)
                loss = F.kl_div(
                    F.log_softmax(output, dim=1), soft_label, reduction="batchmean"
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return

    @staticmethod
    def evaulate(
        model: torch.nn.Module, test_loader: DataLoader, device: str
    ) -> tuple[float, float]:
        model.to(device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                correct = torch.sum(predicted.eq(labels)).item()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += int(correct)
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def get_summary(self) -> dict[str, float]:
        server_loss, server_acc = DSFLServerHandler.evaulate(
            self.model,
            self.dataset.get_dataloader(
                type_="test",
                cid=None,
                batch_size=self.kd_batch_size,
            ),
            self.device,
        )
        client_loss = sum(m["loss"] for m in self.metadata_list) / len(
            self.metadata_list
        )
        client_acc = sum(m["acc"] for m in self.metadata_list) / len(self.metadata_list)
        return {
            "server_acc": server_acc,
            "server_loss": server_loss,
            "client_acc": client_acc,
            "client_loss": client_loss,
        }

    def downlink_package(self) -> DSFLDownlinkPackage:
        next_indices = self.get_next_indices()
        return DSFLDownlinkPackage(
            self.global_soft_labels, self.global_indices, next_indices
        )


@dataclass
class DSFLDiskSharedData:
    model_selector: DSFLModelSelector
    model_name: str
    dataset: DSFLPartitionedDataset
    epochs: int
    batch_size: int
    lr: float
    kd_epochs: int
    kd_batch_size: int
    kd_lr: float
    cid: int
    seed: int
    payload: DSFLDownlinkPackage
    state_path: Path


@dataclass
class DSFLClientState:
    random: RandomState
    model: Mapping[str, torch.Tensor]


class DSFLParallelClientTrainer(
    ParallelClientTrainer[DSFLUplinkPackage, DSFLDownlinkPackage, DSFLDiskSharedData]
):
    def __init__(
        self,
        model_selector: DSFLModelSelector,
        model_name: str,
        share_dir: Path,
        state_dir: Path,
        dataset: DSFLPartitionedDataset,
        device: str,
        num_clients: int,
        epochs: int,
        batch_size: int,
        lr: float,
        kd_epochs: int,
        kd_batch_size: int,
        kd_lr: float,
        seed: int,
        num_parallels: int,
    ) -> None:
        super().__init__(num_parallels, share_dir, device)
        self.model_selector = model_selector
        self.model_name = model_name
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.kd_epochs = kd_epochs
        self.kd_batch_size = kd_batch_size
        self.kd_lr = kd_lr
        self.device = device
        self.num_clients = num_clients
        self.seed = seed

        if self.device == "cuda":
            self.device_count = torch.cuda.device_count()

    @staticmethod
    def process_client(path: Path, device: str) -> Path:
        data = torch.load(path, weights_only=False)
        assert isinstance(data, DSFLDiskSharedData)

        state: DSFLClientState | None = None
        if data.state_path.exists():
            state = torch.load(data.state_path, weights_only=False)
            assert isinstance(state, DSFLClientState)
            RandomState.set_random_state(state.random)
        else:
            seed_everything(data.seed, device=device)

        model = data.model_selector.select_model(data.model_name)

        if state is not None:
            model.load_state_dict(state.model)

        # Distill
        openset = data.dataset.get_dataset(type_="open", cid=None)
        if data.payload.indices is not None and data.payload.soft_labels is not None:
            global_soft_labels = list(torch.unbind(data.payload.soft_labels, dim=0))
            global_indices = data.payload.indices.tolist()
            DSFLServerHandler.distill(
                model=model,
                dataset=data.dataset,
                global_soft_labels=global_soft_labels,
                global_indices=global_indices,
                kd_epochs=data.kd_epochs,
                kd_batch_size=data.kd_batch_size,
                kd_lr=data.kd_lr,
                device=device,
            )

        # Train
        train_loader = data.dataset.get_dataloader(
            type_="train",
            cid=data.cid,
            batch_size=data.batch_size,
        )
        DSFLParallelClientTrainer.train(
            model=model,
            train_loader=train_loader,
            device=device,
            epochs=data.epochs,
            lr=data.lr,
        )

        # Predict
        open_loader = DataLoader(
            Subset(openset, data.payload.next_indices.tolist()),
            batch_size=data.batch_size,
        )
        soft_labels = DSFLParallelClientTrainer.predict(
            model=model,
            open_loader=open_loader,
            device=device,
        )

        # Evaluate
        val_loader = data.dataset.get_dataloader(
            type_="val",
            cid=data.cid,
            batch_size=data.batch_size,
        )
        loss, acc = DSFLServerHandler.evaulate(
            model=model,
            test_loader=val_loader,
            device=device,
        )

        package = DSFLUplinkPackage(
            soft_labels=torch.stack(soft_labels),
            indices=data.payload.next_indices,
            metadata={"loss": loss, "acc": acc},
        )

        torch.save(package, path)
        state = DSFLClientState(
            random=RandomState.get_random_state(device=device),
            model=model.state_dict(),
        )
        torch.save(state, data.state_path)
        return path

    @staticmethod
    def train(
        model: torch.nn.Module,
        train_loader: DataLoader,
        device: str,
        epochs: int,
        lr: float,
    ) -> None:
        model.to(device)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return

    @staticmethod
    def predict(
        model: torch.nn.Module, open_loader: DataLoader, device: str
    ) -> list[torch.Tensor]:
        model.to(device)
        model.eval()

        soft_labels = []
        with torch.no_grad():
            for data in open_loader:
                data = data.to(device)

                output = model(data)
                soft_label = F.softmax(output, dim=1)

                soft_labels.extend([sl.detach().cpu() for sl in soft_label])

        return soft_labels

    def get_shared_data(
        self, cid: int, payload: DSFLDownlinkPackage
    ) -> DSFLDiskSharedData:
        data = DSFLDiskSharedData(
            model_selector=self.model_selector,
            model_name=self.model_name,
            dataset=self.dataset,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            kd_epochs=self.kd_epochs,
            kd_batch_size=self.kd_batch_size,
            kd_lr=self.kd_lr,
            cid=cid,
            seed=self.seed,
            payload=payload,
            state_path=self.state_dir.joinpath(f"{cid}.pt"),
        )
        return data

    def uplink_package(self) -> list[DSFLUplinkPackage]:
        package = deepcopy(self.cache)
        self.cache: list[DSFLUplinkPackage] = []
        return package
