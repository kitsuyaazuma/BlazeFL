import random
from dataclasses import dataclass
from logging import Logger
from pathlib import Path

import torch
import torch.multiprocessing as mp
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from blazefl.core import (
    ModelSelector,
    ParallelClientTrainer,
    PartitionedDataset,
    SerialClientTrainer,
    ServerHandler,
    SharedData,
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
                data.to(self.device)
                target.to(self.device)

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


class FedAvgSharedMemoryData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_selector: ModelSelector
    model_name: str
    dataset: PartitionedDataset
    epochs: int
    batch_size: int
    lr: float
    device: str
    cid: int


class FedAvgDiskData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model_parameters: torch.Tensor


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
        shared_data: SharedData[FedAvgSharedMemoryData, FedAvgDiskData],
    ) -> FedAvgUplinkPackage:
        shared_memory_data = shared_data.get_shared_memory_data()
        disk_data = shared_data.get_disk_data()
        model = shared_memory_data.model_selector.select_model(
            shared_memory_data.model_name
        )

        deserialize_model(model, disk_data.model_parameters)
        model.to(shared_memory_data.device)
        model.train()
        train_loader = shared_memory_data.dataset.get_dataloader(
            type_="train",
            cid=shared_memory_data.cid,
            batch_size=shared_memory_data.batch_size,
        )
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=shared_memory_data.lr)

        data_size = 0
        for _ in range(shared_memory_data.epochs):
            for data, target in train_loader:
                data.to(shared_memory_data.device)
                target.to(shared_memory_data.device)

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
    ) -> SharedData[FedAvgSharedMemoryData, FedAvgDiskData]:
        shared_memory_data = FedAvgSharedMemoryData(
            model_selector=self.model_selector,
            model_name=self.model_name,
            dataset=self.dataset,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=f"cuda:{cid % self.device_count}"
            if self.device == "cuda"
            else self.device,
            cid=cid,
        )
        disk_data = FedAvgDiskData(model_parameters=payload.model_parameters)
        shared_data = SharedData(
            shared_memory_data=shared_memory_data,
            disk_data=disk_data,
            disk_path=self.tmp_dir.joinpath(f"{cid}.pt"),
        )
        return shared_data

    def local_process(
        self, payload: FedAvgDownlinkPackage, cid_list: list[int]
    ) -> None:
        pool = mp.Pool(processes=self.num_parallels)
        jobs = []
        for cid in cid_list:
            client_shared_data = self.get_shared_data(cid, payload).share()
            jobs.append(pool.apply_async(self.process_client, (client_shared_data,)))

        for job in tqdm(jobs, desc="Client", leave=False):
            result = job.get()
            assert isinstance(result, FedAvgUplinkPackage)
            self.cache.append(result)

    def uplink_package(self) -> list[FedAvgUplinkPackage]:
        return self.cache
