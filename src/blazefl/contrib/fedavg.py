import random
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from blazefl.core import (
    ModelSelector,
    ParallelClientTrainer,
    PartitionedDataset,
    SerialClientTrainer,
    ServerHandler,
)
from blazefl.utils import (
    RandomState,
    deserialize_model,
    seed_everything,
    serialize_model,
)


@dataclass
class FedAvgUplinkPackage:
    """
    Data structure representing the uplink package sent from clients to the server
    in the Federated Averaging algorithm.

    Attributes:
        model_parameters (torch.Tensor): Serialized model parameters from the client.
        data_size (int): Number of data samples used in the client's training.
        metadata (dict | None): Optional metadata, such as evaluation metrics.
    """

    model_parameters: torch.Tensor
    data_size: int
    metadata: dict | None = None


@dataclass
class FedAvgDownlinkPackage:
    """
    Data structure representing the downlink package sent from the server to clients
    in the Federated Averaging algorithm.

    Attributes:
        model_parameters (torch.Tensor): Serialized global model parameters to be
        distributed to clients.
    """

    model_parameters: torch.Tensor


class FedAvgServerHandler(ServerHandler):
    """
    Server-side handler for the Federated Averaging (FedAvg) algorithm.

    Manages the global model, coordinates client sampling, aggregates client updates,
    and controls the training process across multiple rounds.

    Attributes:
        model (torch.nn.Module): The global model being trained.
        dataset (PartitionedDataset): Dataset partitioned across clients.
        global_round (int): Total number of federated learning rounds.
        num_clients (int): Total number of clients in the federation.
        sample_ratio (float): Fraction of clients to sample in each round.
        device (str): Device to run the model on ('cpu' or 'cuda').
        client_buffer_cache (list[FedAvgUplinkPackage]): Cache for storing client
        updates before aggregation.
        num_clients_per_round (int): Number of clients sampled per round.
        round (int): Current training round.
    """

    def __init__(
        self,
        model_selector: ModelSelector,
        model_name: str,
        dataset: PartitionedDataset,
        global_round: int,
        num_clients: int,
        sample_ratio: float,
        device: str,
    ) -> None:
        """
        Initialize the FedAvgServerHandler.

        Args:
            model_selector (ModelSelector): Selector for initializing the model.
            model_name (str): Name of the model to be used.
            dataset (PartitionedDataset): Dataset partitioned across clients.
            global_round (int): Total number of federated learning rounds.
            num_clients (int): Total number of clients in the federation.
            sample_ratio (float): Fraction of clients to sample in each round.
            device (str): Device to run the model on ('cpu' or 'cuda').
        """
        self.model = model_selector.select_model(model_name)
        self.dataset = dataset
        self.global_round = global_round
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.device = device

        self.client_buffer_cache: list[FedAvgUplinkPackage] = []
        self.num_clients_per_round = int(self.num_clients * self.sample_ratio)
        self.round = 0

    def sample_clients(self) -> list[int]:
        """
        Randomly sample a subset of clients for the current training round.

        Returns:
            list[int]: Sorted list of sampled client IDs.
        """
        sampled_clients = random.sample(
            range(self.num_clients), self.num_clients_per_round
        )

        return sorted(sampled_clients)

    def if_stop(self) -> bool:
        """
        Check if the training process should stop.

        Returns:
            bool: True if the current round exceeds or equals the total number of
            global rounds; False otherwise.
        """
        return self.round >= self.global_round

    def load(self, payload: FedAvgUplinkPackage) -> bool:
        """
        Load a client's uplink package into the server's buffer and perform a global
        update if all expected packages for the round are received.

        Args:
            payload (FedAvgUplinkPackage): Uplink package from a client.

        Returns:
            bool: True if a global update was performed; False otherwise.
        """
        self.client_buffer_cache.append(payload)

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1
            self.client_buffer_cache = []
            return True
        else:
            return False

    def global_update(self, buffer: list[FedAvgUplinkPackage]) -> None:
        """
        Aggregate client updates and update the global model parameters.

        Args:
            buffer (list[FedAvgUplinkPackage]): List of uplink packages from clients.
        """
        parameters_list = [ele.model_parameters for ele in buffer]
        weights_list = [ele.data_size for ele in buffer]
        serialized_parameters = self.aggregate(parameters_list, weights_list)
        deserialize_model(self.model, serialized_parameters)

    @staticmethod
    def aggregate(
        parameters_list: list[torch.Tensor], weights_list: list[int]
    ) -> torch.Tensor:
        """
        Aggregate model parameters from multiple clients using weighted averaging.

        Args:
            parameters_list (list[torch.Tensor]): List of serialized model parameters
            from clients.
            weights_list (list[int]): List of data sizes corresponding to each client's
            parameters.

        Returns:
            torch.Tensor: Aggregated model parameters.
        """
        parameters = torch.stack(parameters_list, dim=-1)
        weights = torch.tensor(weights_list)
        weights = weights / torch.sum(weights)

        serialized_parameters = torch.sum(parameters * weights, dim=-1)

        return serialized_parameters

    def downlink_package(self) -> FedAvgDownlinkPackage:
        """
        Create a downlink package containing the current global model parameters to
        send to clients.

        Returns:
            FedAvgDownlinkPackage: Downlink package with serialized model parameters.
        """
        model_parameters = serialize_model(self.model)
        return FedAvgDownlinkPackage(model_parameters)


class FedAvgSerialClientTrainer(
    SerialClientTrainer[FedAvgUplinkPackage, FedAvgDownlinkPackage]
):
    """
    Serial client trainer for the Federated Averaging (FedAvg) algorithm.

    This trainer processes clients sequentially, training and evaluating a local model
    for each client based on the server-provided model parameters.

    Attributes:
        model (torch.nn.Module): The client's local model.
        dataset (PartitionedDataset): Dataset partitioned across clients.
        device (str): Device to run the model on ('cpu' or 'cuda').
        num_clients (int): Total number of clients in the federation.
        epochs (int): Number of local training epochs per client.
        batch_size (int): Batch size for local training.
        lr (float): Learning rate for the optimizer.
        cache (list[FedAvgUplinkPackage]): Cache to store uplink packages for the
        server.
    """

    def __init__(
        self,
        model_selector: ModelSelector,
        model_name: str,
        dataset: PartitionedDataset,
        device: str,
        num_clients: int,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        """
        Initialize the FedAvgSerialClientTrainer.

        Args:
            model_selector (ModelSelector): Selector for initializing the local model.
            model_name (str): Name of the model to be used.
            dataset (PartitionedDataset): Dataset partitioned across clients.
            device (str): Device to run the model on ('cpu' or 'cuda').
            num_clients (int): Total number of clients in the federation.
            epochs (int): Number of local training epochs per client.
            batch_size (int): Batch size for local training.
            lr (float): Learning rate for the optimizer.
        """
        self.model = model_selector.select_model(model_name)
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
        """
        Train and evaluate the model for each client in the given list.

        Args:
            payload (FedAvgDownlinkPackage): Downlink package with global model
            parameters.
            cid_list (list[int]): List of client IDs to process.

        Returns:
            None
        """
        model_parameters = payload.model_parameters
        for cid in tqdm(cid_list, desc="Client", leave=False):
            data_loader = self.dataset.get_dataloader(
                type_="train", cid=cid, batch_size=self.batch_size
            )
            pack = self.train(model_parameters, data_loader)
            val_loader = self.dataset.get_dataloader(
                type_="val", cid=cid, batch_size=self.batch_size
            )
            loss, acc = self.evaluate(val_loader)
            pack.metadata = {"loss": loss, "acc": acc}
            self.cache.append(pack)

    def train(
        self, model_parameters: torch.Tensor, train_loader: DataLoader
    ) -> FedAvgUplinkPackage:
        """
        Train the local model on the given training data loader.

        Args:
            model_parameters (torch.Tensor): Global model parameters to initialize the
            local model.
            train_loader (DataLoader): DataLoader for the training data.

        Returns:
            FedAvgUplinkPackage: Uplink package containing updated model parameters and
            data size.
        """
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

    def evaluate(self, test_loader: DataLoader) -> tuple[float, float]:
        """
        Evaluate the local model on the given test data loader.

        Args:
            test_loader (DataLoader): DataLoader for the evaluation data.

        Returns:
            tuple[float, float]: A tuple containing the average loss and accuracy.
        """
        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                correct = torch.sum(predicted.eq(labels)).item()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += int(correct)
                total_samples += batch_size

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def uplink_package(self) -> list[FedAvgUplinkPackage]:
        """
        Retrieve the uplink packages for transmission to the server.

        Returns:
            list[FedAvgUplinkPackage]: A list of uplink packages.
        """
        package = deepcopy(self.cache)
        self.cache = []
        return package


@dataclass
class FedAvgDiskSharedData:
    """
    Data structure representing shared data for parallel client training
    in the Federated Averaging (FedAvg) algorithm.

    This structure is used to store all necessary information for each client
    to perform local training in a parallelized setting.

    Attributes:
        model_selector (ModelSelector): Selector for initializing the local model.
        model_name (str): Name of the model to be used.
        dataset (PartitionedDataset): Dataset partitioned across clients.
        epochs (int): Number of local training epochs per client.
        batch_size (int): Batch size for local training.
        lr (float): Learning rate for the optimizer.
        cid (int): Client ID.
        seed (int): Seed for reproducibility.
        payload (FedAvgDownlinkPackage): Downlink package with global model parameters.
        state_path (Path): Path to save the client's random state.
    """

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


class FedAvgParallelClientTrainer(
    ParallelClientTrainer[
        FedAvgUplinkPackage, FedAvgDownlinkPackage, FedAvgDiskSharedData
    ]
):
    """
    Parallel client trainer for the Federated Averaging (FedAvg) algorithm.

    This trainer handles the parallelized training and evaluation of local models
    across multiple clients, distributing tasks to different processes or devices.

    Attributes:
        model_selector (ModelSelector): Selector for initializing the local model.
        model_name (str): Name of the model to be used.
        share_dir (Path): Directory to store shared data files between processes.
        state_dir (Path): Directory to save random states for reproducibility.
        dataset (PartitionedDataset): Dataset partitioned across clients.
        device (str): Device to run the models on ('cpu' or 'cuda').
        num_clients (int): Total number of clients in the federation.
        epochs (int): Number of local training epochs per client.
        batch_size (int): Batch size for local training.
        lr (float): Learning rate for the optimizer.
        seed (int): Seed for reproducibility.
        num_parallels (int): Number of parallel processes for training.
        device_count (int | None): Number of CUDA devices available (if using GPU).
    """

    def __init__(
        self,
        model_selector: ModelSelector,
        model_name: str,
        share_dir: Path,
        state_dir: Path,
        dataset: PartitionedDataset,
        device: str,
        num_clients: int,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        num_parallels: int,
    ) -> None:
        """
        Initialize the FedAvgParalleClientTrainer.

        Args:
            model_selector (ModelSelector): Selector for initializing the local model.
            model_name (str): Name of the model to be used.
            share_dir (Path): Directory to store shared data files between processes.
            state_dir (Path): Directory to save random states for reproducibility.
            dataset (PartitionedDataset): Dataset partitioned across clients.
            device (str): Device to run the models on ('cpu' or 'cuda').
            num_clients (int): Total number of clients in the federation.
            epochs (int): Number of local training epochs per client.
            batch_size (int): Batch size for local training.
            lr (float): Learning rate for the optimizer.
            seed (int): Seed for reproducibility.
            num_parallels (int): Number of parallel processes for training.
        """
        super().__init__(num_parallels, share_dir, device)
        self.model_selector = model_selector
        self.model_name = model_name
        self.state_dir = state_dir
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.num_clients = num_clients
        self.seed = seed

    @staticmethod
    def process_client(path: Path, device: str) -> Path:
        """
        Process a single client's local training and evaluation.

        This method is executed by a parallel process and handles data loading,
        training, evaluation, and saving results to a shared file.

        Args:
            path (Path): Path to the shared data file containing client-specific
            information.
            device (str): Device to use for processing.

        Returns:
            Path: Path to the file with the processed results.
        """
        data = torch.load(path, weights_only=False)
        assert isinstance(data, FedAvgDiskSharedData)

        if data.state_path.exists():
            state = torch.load(data.state_path, weights_only=False)
            assert isinstance(state, RandomState)
            RandomState.set_random_state(state)
        else:
            seed_everything(data.seed, device=device)

        model = data.model_selector.select_model(data.model_name)
        train_loader = data.dataset.get_dataloader(
            type_="train",
            cid=data.cid,
            batch_size=data.batch_size,
        )
        package = FedAvgParallelClientTrainer.train(
            model=model,
            model_parameters=data.payload.model_parameters,
            train_loader=train_loader,
            device=device,
            epochs=data.epochs,
            lr=data.lr,
        )
        val_loader = data.dataset.get_dataloader(
            type_="val",
            cid=data.cid,
            batch_size=data.batch_size,
        )
        loss, acc = FedAvgParallelClientTrainer.evaulate(
            model=model,
            test_loader=val_loader,
            device=device,
        )
        package.metadata = {"loss": loss, "acc": acc}
        torch.save(package, path)
        torch.save(RandomState.get_random_state(device=device), data.state_path)
        return path

    @staticmethod
    def train(
        model: torch.nn.Module,
        model_parameters: torch.Tensor,
        train_loader: DataLoader,
        device: str,
        epochs: int,
        lr: float,
    ) -> FedAvgUplinkPackage:
        """
        Train the model for a single client.

        Args:
            model (torch.nn.Module): The model to train.
            model_parameters (torch.Tensor): Initial global model parameters.
            train_loader (DataLoader): DataLoader for the training data.
            device (str): Device to run the training on.
            epochs (int): Number of local training epochs.
            lr (float): Learning rate for the optimizer.

        Returns:
            FedAvgUplinkPackage: Uplink package containing updated model parameters
            and data size.
        """
        model.to(device)
        deserialize_model(model, model_parameters)
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        data_size = 0
        for _ in range(epochs):
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

    @staticmethod
    def evaulate(
        model: torch.nn.Module, test_loader: DataLoader, device: str
    ) -> tuple[float, float]:
        """
        Evaluate the model for a single client.

        Args:
            model (torch.nn.Module): The model to evaluate.
            test_loader (DataLoader): DataLoader for the evaluation data.
            device (str): Device to run the evaluation on.

        Returns:
            tuple[float, float]: Average loss and accuracy.
        """
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

    def get_shared_data(
        self, cid: int, payload: FedAvgDownlinkPackage
    ) -> FedAvgDiskSharedData:
        """
        Generate the shared data for a specific client.

        Args:
            cid (int): Client ID.
            payload (FedAvgDownlinkPackage): Downlink package with global model
            parameters.

        Returns:
            FedAvgDiskSharedData: Shared data structure for the client.
        """
        data = FedAvgDiskSharedData(
            model_selector=self.model_selector,
            model_name=self.model_name,
            dataset=self.dataset,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            cid=cid,
            seed=self.seed,
            payload=payload,
            state_path=self.state_dir.joinpath(f"{cid}.pt"),
        )
        return data

    def uplink_package(self) -> list[FedAvgUplinkPackage]:
        """
        Retrieve the uplink packages for transmission to the server.

        Returns:
            list[FedAvgUplinkPackage]: A list of uplink packages.
        """
        package = deepcopy(self.cache)
        self.cache = []
        return package
