import random
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from blazefl.core import (
    ModelSelector,
    ParallelClientTrainer,
    PartitionedDataset,
    SerialClientTrainer,
    ServerHandler,
    SharedDisk,
)
from blazefl.utils.serialize import deserialize_model, serialize_model


@dataclass
class FedAvgUplinkPackage:
    model_parameters: torch.Tensor
    data_size: int


@dataclass
class FedAvgDownlinkPackage:
    model_parameters: torch.Tensor


class FedAvgServerHandler(ServerHandler):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: PartitionedDataset,
        global_round: int,
        num_clients: int,
        sample_ratio: float,
        device: str,
        logger: Logger,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.global_round = global_round
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.device = device
        self.logger = logger

        self.client_buffer_cache: list[FedAvgUplinkPackage] = []
        self.num_clients_per_round = int(self.num_clients * self.sample_ratio)
        self.round = 0

    def sample_clients(self) -> list[int]:
        sampled_clients = random.sample(
            range(self.num_clients), self.num_clients_per_round
        )

        return sorted(sampled_clients)

    def if_stop(self) -> bool:
        return self.round >= self.global_round

    def load(self, payload: FedAvgUplinkPackage) -> bool:
        self.client_buffer_cache.append(payload)

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1
            self.client_buffer_cache = []
            return True
        else:
            return False

    def global_update(self, buffer: list[FedAvgUplinkPackage]) -> None:
        parameters_list = [ele.model_parameters for ele in buffer]
        weights_list = [ele.data_size for ele in buffer]
        serialized_parameters = self.aggregate(parameters_list, weights_list)
        deserialize_model(self.model, serialized_parameters)

    @staticmethod
    def aggregate(
        parameters_list: list[torch.Tensor], weights_list: list[int]
    ) -> torch.Tensor:
        parameters = torch.stack(parameters_list, dim=-1)
        weights = torch.tensor(weights_list)
        weights = weights / torch.sum(weights)

        serialized_parameters = torch.sum(parameters * weights, dim=-1)

        return serialized_parameters

    def downlink_package(self) -> FedAvgDownlinkPackage:
        model_parameters = serialize_model(self.model)
        return FedAvgDownlinkPackage(model_parameters)


class FedAvgSerialClientTrainer(SerialClientTrainer):
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: PartitionedDataset,
        device: str,
        num_clients: int,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        self.model = model
        self.dataset = dataset
        self.device = device
        self.num_clients = num_clients
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.model.to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.cache: list[FedAvgUplinkPackage] = []

    def local_process(
        self, payload: FedAvgDownlinkPackage, cid_list: list[int]
    ) -> None:
        model_parameters = payload.model_parameters
        for cid in tqdm(cid_list, desc="Client", leave=False):
            data_loader = self.dataset.get_dataloader(
                type_="train", cid=cid, batch_size=self.batch_size
            )
            pack = self.train(model_parameters, data_loader)
            self.cache.append(pack)

    def train(self, model_parameters, train_loader) -> FedAvgUplinkPackage:
        deserialize_model(self.model, model_parameters)
        self.model.train()

        data_size = 0
        for _ in range(self.epochs):
            for data, target in train_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                data_size += len(target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        model_parameters = serialize_model(self.model)

        return FedAvgUplinkPackage(model_parameters, data_size)

    def uplink_package(self) -> list[FedAvgUplinkPackage]:
        return self.cache


@dataclass
class FedAvgDiskSharedData:
    model_selector: ModelSelector
    model_name: str
    model_parameters: torch.Tensor
    dataset: PartitionedDataset
    epochs: int
    batch_size: int
    lr: float
    device: str
    cid: int


class FedAvgParalleClientTrainer(ParallelClientTrainer):
    def __init__(
        self,
        model_selector: ModelSelector,
        model_name: str,
        tmp_dir: Path,
        dataset: PartitionedDataset,
        device: str,
        num_clients: int,
        epochs: int,
        batch_size: int,
        lr: float,
        num_parallels: int,
    ) -> None:
        self.model_selector = model_selector
        self.model_name = model_name
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.tmp_dir = tmp_dir
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.num_clients = num_clients
        self.num_parallels = num_parallels

        self.cache: list[FedAvgUplinkPackage] = []
        if self.device == "cuda":
            self.device_count = torch.cuda.device_count()

    @staticmethod
    def process_client(
        shared_disk: SharedDisk[FedAvgDiskSharedData],
    ) -> FedAvgUplinkPackage:
        shared_data = shared_disk.get_data()
        device = shared_data.device
        model = shared_data.model_selector.select_model(shared_data.model_name).to(
            device
        )
        deserialize_model(model, shared_data.model_parameters)
        train_loader = shared_data.dataset.get_dataloader(
            type_="train",
            cid=shared_data.cid,
            batch_size=shared_data.batch_size,
        )
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=shared_data.lr)

        model.train()
        data_size = 0
        for _ in range(shared_data.epochs):
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = criterion(output, target)

                data_size += len(target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model_parameters = serialize_model(model)
        return FedAvgUplinkPackage(model_parameters, data_size)

    def get_shared_data(
        self, cid: int, payload: FedAvgDownlinkPackage
    ) -> SharedDisk[FedAvgDiskSharedData]:
        data = FedAvgDiskSharedData(
            model_selector=self.model_selector,
            model_name=self.model_name,
            model_parameters=payload.model_parameters,
            dataset=self.dataset,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=f"cuda:{cid % self.device_count}"
            if self.device == "cuda"
            else self.device,
            cid=cid,
        )
        disk_shared_data = SharedDisk(
            data=data,
            path=self.tmp_dir.joinpath(f"{cid}.pkl"),
        )
        return disk_shared_data

    def local_process(
        self, payload: FedAvgDownlinkPackage, cid_list: list[int]
    ) -> None:
        pool = mp.Pool(processes=self.num_parallels)
        jobs = []
        for cid in cid_list:
            shared_disk = self.get_shared_data(cid, payload)
            shared_disk.share()
            jobs.append(pool.apply_async(self.process_client, (shared_disk,)))

        for job in tqdm(jobs, desc="Client", leave=False):
            result = job.get()
            assert isinstance(result, FedAvgUplinkPackage)
            self.cache.append(result)

    def uplink_package(self) -> list[FedAvgUplinkPackage]:
        return self.cache
