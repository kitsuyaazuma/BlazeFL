import torch
from _typeshed import Incomplete
from blazefl.core import ModelSelector as ModelSelector, ParallelClientTrainer as ParallelClientTrainer, PartitionedDataset as PartitionedDataset, SerialClientTrainer as SerialClientTrainer, ServerHandler as ServerHandler
from blazefl.utils import RandomState as RandomState, deserialize_model as deserialize_model, seed_everything as seed_everything, serialize_model as serialize_model
from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader

@dataclass
class FedAvgUplinkPackage:
    model_parameters: torch.Tensor
    data_size: int
    metadata: dict | None = ...

@dataclass
class FedAvgDownlinkPackage:
    model_parameters: torch.Tensor

class FedAvgServerHandler(ServerHandler):
    model: Incomplete
    dataset: Incomplete
    global_round: Incomplete
    num_clients: Incomplete
    sample_ratio: Incomplete
    device: Incomplete
    client_buffer_cache: Incomplete
    num_clients_per_round: Incomplete
    round: int
    def __init__(self, model_selector: ModelSelector, model_name: str, dataset: PartitionedDataset, global_round: int, num_clients: int, sample_ratio: float, device: str) -> None: ...
    def sample_clients(self) -> list[int]: ...
    def if_stop(self) -> bool: ...
    def load(self, payload: FedAvgUplinkPackage) -> bool: ...
    def global_update(self, buffer: list[FedAvgUplinkPackage]) -> None: ...
    @staticmethod
    def aggregate(parameters_list: list[torch.Tensor], weights_list: list[int]) -> torch.Tensor: ...
    def downlink_package(self) -> FedAvgDownlinkPackage: ...

class FedAvgSerialClientTrainer(SerialClientTrainer[FedAvgUplinkPackage, FedAvgDownlinkPackage]):
    model: Incomplete
    dataset: Incomplete
    device: Incomplete
    num_clients: Incomplete
    epochs: Incomplete
    batch_size: Incomplete
    lr: Incomplete
    optimizer: Incomplete
    criterion: Incomplete
    cache: Incomplete
    def __init__(self, model_selector: ModelSelector, model_name: str, dataset: PartitionedDataset, device: str, num_clients: int, epochs: int, batch_size: int, lr: float) -> None: ...
    def local_process(self, payload: FedAvgDownlinkPackage, cid_list: list[int]) -> None: ...
    def train(self, model_parameters: torch.Tensor, train_loader: DataLoader) -> FedAvgUplinkPackage: ...
    def evaluate(self, test_loader: DataLoader) -> tuple[float, float]: ...
    def uplink_package(self) -> list[FedAvgUplinkPackage]: ...

@dataclass
class FedAvgDiskSharedData:
    model_selector: ModelSelector
    model_name: str
    dataset: PartitionedDataset
    epochs: int
    batch_size: int
    lr: float
    cid: int
    seed: int
    payload: FedAvgDownlinkPackage
    state_path: Path

class FedAvgParallelClientTrainer(ParallelClientTrainer[FedAvgUplinkPackage, FedAvgDownlinkPackage, FedAvgDiskSharedData]):
    model_selector: Incomplete
    model_name: Incomplete
    state_dir: Incomplete
    dataset: Incomplete
    epochs: Incomplete
    batch_size: Incomplete
    lr: Incomplete
    device: Incomplete
    num_clients: Incomplete
    seed: Incomplete
    def __init__(self, model_selector: ModelSelector, model_name: str, share_dir: Path, state_dir: Path, dataset: PartitionedDataset, device: str, num_clients: int, epochs: int, batch_size: int, lr: float, seed: int, num_parallels: int) -> None: ...
    @staticmethod
    def process_client(path: Path, device: str) -> Path: ...
    @staticmethod
    def train(model: torch.nn.Module, model_parameters: torch.Tensor, train_loader: DataLoader, device: str, epochs: int, lr: float) -> FedAvgUplinkPackage: ...
    @staticmethod
    def evaulate(model: torch.nn.Module, test_loader: DataLoader, device: str) -> tuple[float, float]: ...
    def get_shared_data(self, cid: int, payload: FedAvgDownlinkPackage) -> FedAvgDiskSharedData: ...
    cache: Incomplete
    def uplink_package(self) -> list[FedAvgUplinkPackage]: ...
