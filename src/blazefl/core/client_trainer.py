import multiprocessing as mp
import signal
from abc import ABC, abstractmethod
from multiprocessing.pool import ApplyResult
from pathlib import Path
from typing import Generic, TypeVar

import torch
from tqdm import tqdm

UplinkPackage = TypeVar("UplinkPackage")
DownlinkPackage = TypeVar("DownlinkPackage")


class SerialClientTrainer(ABC, Generic[UplinkPackage, DownlinkPackage]):
    """
    Abstract base class for serial client training in federated learning.

    This class defines the interface for training clients in a serial manner,
    where each client is processed one after the other.

    Raises:
        NotImplementedError: If the methods are not implemented in a subclass.
    """

    @abstractmethod
    def uplink_package(self) -> list[UplinkPackage]:
        """
        Prepare the data package to be sent from the client to the server.

        Returns:
            list[UplinkPackage]: A list of data packages prepared for uplink
            transmission.
        """
        ...

    @abstractmethod
    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None:
        """
        Process the downlink payload from the server for a list of client IDs.

        Args:
            payload (DownlinkPackage): The data package received from the server.
            cid_list (list[int]): A list of client IDs to process.

        Returns:
            None
        """
        ...


DiskSharedData = TypeVar("DiskSharedData")


class ParallelClientTrainer(
    SerialClientTrainer[UplinkPackage, DownlinkPackage],
    Generic[UplinkPackage, DownlinkPackage, DiskSharedData],
):
    """
    Abstract base class for parallel client training in federated learning.

    This class extends SerialClientTrainer to enable parallel processing of clients,
    allowing multiple clients to be trained concurrently.

    Attributes:
        num_parallels (int): Number of parallel processes to use for client training.
        share_dir (Path): Directory path for sharing data between processes.
        cache (list[UplinkPackage]): Cache to store uplink packages from clients.

    Raises:
        NotImplementedError: If the abstract methods are not implemented in a subclass.
    """

    def __init__(self, num_parallels: int, share_dir: Path, device: str) -> None:
        """
        Initialize the ParallelClientTrainer with parallelism settings.

        Args:
            num_parallels (int): Number of parallel processes to use.
            share_dir (Path): Directory path for sharing data between processes.
            device (str): Device to use for processing clients.
        """
        self.num_parallels = num_parallels
        self.share_dir = share_dir
        self.share_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        if self.device == "cuda":
            self.device_count = torch.cuda.device_count()
        self.cache: list[UplinkPackage] = []

    @abstractmethod
    def get_shared_data(self, cid: int, payload: DownlinkPackage) -> DiskSharedData:
        """
        Retrieve shared data for a given client ID and payload.

        Args:
            cid (int): Client ID.
            payload (DownlinkPackage): The data package received from the server.

        Returns:
            DiskSharedData: The shared data associated with the client ID and payload.
        """
        ...

    def get_client_device(self, cid: int) -> str:
        """
        Retrieve the device to use for processing a given client.

        Args:
            cid (int): Client ID.

        Returns:
            str: The device to use for processing the client.
        """
        if self.device == "cuda":
            return f"cuda:{cid % self.device_count}"
        return self.device

    @staticmethod
    @abstractmethod
    def process_client(path: Path, device: str) -> Path:
        """
        Process a single client based on the provided path.

        Args:
            path (Path): Path to the client's data file.
            device (str): Device to use for processing.

        Returns:
            Path: Path to the processed client's data file.
        """
        ...

    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None:
        """
        Manage the parallel processing of clients.

        This method distributes the processing of multiple clients across
        parallel processes, handling data saving, loading, and caching.

        Args:
            payload (DownlinkPackage): The data package received from the server.
            cid_list (list[int]): A list of client IDs to process.

        Returns:
            None
        """
        with mp.Pool(
            processes=self.num_parallels,
            initializer=signal.signal,
            initargs=(signal.SIGINT, signal.SIG_IGN),
        ) as pool:
            jobs: list[ApplyResult] = []
            for cid in cid_list:
                path = self.share_dir.joinpath(f"{cid}.pkl")
                data = self.get_shared_data(cid, payload)
                device = self.get_client_device(cid)
                torch.save(data, path)
                jobs.append(pool.apply_async(self.process_client, (path, device)))

            for job in tqdm(jobs, desc="Client", leave=False):
                path = job.get()
                assert isinstance(path, Path)
                package = torch.load(path, weights_only=False)
                self.cache.append(package)
