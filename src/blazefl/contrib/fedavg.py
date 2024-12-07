from logging import Logger
import torch
import random
from dataclasses import dataclass
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

from blazefl.core import (
    ServerHandler,
    ModelSelector,
    PartitionedDataset,
    SerialClientTrainer,
    ParallelClientTrainer,
)
from blazefl.utils.serialize import serialize_model, deserialize_model
from blazefl.utils.share import set_shared_memory, set_shared_disk, get_shared_disk


@dataclass
class FedAvgPackage:
    model_parameters: torch.Tensor
    data_size: int


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

        self.client_buffer_cache: list[FedAvgPackage] = []
        self.num_clients_per_round = int(self.num_clients * self.sample_ratio)
        self.round = 0

    def sample_clients(self) -> list[int]:
        sampled_clients = random.sample(
            range(self.num_clients), self.num_clients_per_round
        )

        return sorted(sampled_clients)

    def if_stop(self) -> bool:
        return self.round >= self.global_round

    def load(self, payload: FedAvgPackage) -> bool:
        self.client_buffer_cache.append(payload)

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1
            self.client_buffer_cache = []
            return True
        else:
            return False

    def global_update(self, buffer: list[FedAvgPackage]) -> None:
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

    def downlink_package(self) -> FedAvgPackage:
        model_parameters = serialize_model(self.model)
        return FedAvgPackage(model_parameters, 0)


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
        self.cache: list[FedAvgPackage] = []

    def local_process(self, payload: FedAvgPackage, cid_list: list[int]):
        model_parameters = payload.model_parameters
        for cid in tqdm(cid_list, desc="Client", leave=False):
            data_loader = self.dataset.get_dataloader(
                type_="train", cid=cid, batch_size=self.batch_size
            )
            pack = self.train(model_parameters, data_loader)
            self.cache.append(pack)

    def train(self, model_parameters, train_loader) -> FedAvgPackage:
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

        return FedAvgPackage(model_parameters, data_size)

    def uplink_package(self) -> list[FedAvgPackage]:
        return self.cache


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
        self.tmp_dir = tmp_dir
        self.dataset = dataset
        self.device = device
        self.num_clients = num_clients
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_parallels = num_parallels

        self.cache: list[FedAvgPackage] = []
        if self.device == "cuda":
            self.device_count = torch.cuda.device_count()

    @staticmethod
    def process_client(memory_parameters: dict, disk_path: Path) -> tuple[dict, Path]:
        disk_parameters = get_shared_disk(disk_path)["model_parameters"]
        model_parameters = disk_parameters["model_parameters"]
        model_selector = memory_parameters["model_selector"]
        model_name = memory_parameters["model_name"]
        model = model_selector.select_model(model_name)
        device = memory_parameters["device"]
        dataset = memory_parameters["dataset"]
        epochs, batch_size, lr = (
            memory_parameters["epochs"],
            memory_parameters["batch_size"],
            memory_parameters["lr"],
        )

        deserialize_model(model, model_parameters)
        model.to(device)
        model.train()
        train_loader = dataset.get_dataloader(
            type_="train", cid=memory_parameters["cid"], batch_size=batch_size
        )
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)

        data_size = 0
        for _ in range(epochs):
            for data, target in train_loader:
                data.to(device)
                target.to(device)

                output = model(data)
                loss = criterion(output, target)

                data_size += len(target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model_parameters = serialize_model(model)

        memory_parameters["data_size"] = data_size
        disk_parameters = {
            "model_parameters": model_parameters,
        }
        disk_path = memory_parameters["disk_path"].joinpath(
            f"{memory_parameters['cid']}.pt"
        )
        set_shared_disk(disk_parameters, disk_path)
        set_shared_memory(memory_parameters)

        return memory_parameters, disk_path

    def local_process(self, payload: FedAvgPackage, cid_list: list[int]):
        model_parameters = payload.model_parameters

        common_memory_parameters = {}
        disk_parameters = {
            "model_parameters": model_parameters,
        }

        pool = mp.Pool(processes=self.num_parallels)
        jobs = []
        for cid in cid_list:
            memory_parameters = common_memory_parameters.copy()
            memory_parameters["device"] = (
                f"cuda:{self.device_count % self.num_clients}"
                if self.device == "cuda"
                else "cpu"
            )
            set_shared_memory(memory_parameters)
            disk_path = self.tmp_dir.joinpath(f"{cid}.pt")
            set_shared_disk(disk_parameters, disk_path)
            jobs.append(
                pool.apply_async(
                    self.process_client,
                    (
                        memory_parameters,
                        disk_path,
                    ),
                )
            )

        for job in jobs:
            memory_parameters, disk_path = job.get()
            disk_parameters = torch.load(disk_path, weights_only=False)
            self.cache.append(
                FedAvgPackage(
                    disk_parameters["model_parameters"], memory_parameters["data_size"]
                )
            )

    def uplink_package(self) -> list[FedAvgPackage]:
        return self.cache
