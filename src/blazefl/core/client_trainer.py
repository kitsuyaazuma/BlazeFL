import multiprocessing as mp
from abc import ABC, abstractmethod
from multiprocessing.pool import ApplyResult
from pathlib import Path
from typing import Generic, TypeVar

import torch
from tqdm import tqdm

UplinkPackage = TypeVar("UplinkPackage")
DownlinkPackage = TypeVar("DownlinkPackage")


class SerialClientTrainer(ABC, Generic[UplinkPackage, DownlinkPackage]):
    @abstractmethod
    def uplink_package(self) -> list[UplinkPackage]: ...

    @abstractmethod
    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None: ...


DiskSharedData = TypeVar("DiskSharedData")


class ParallelClientTrainer(
    SerialClientTrainer[UplinkPackage, DownlinkPackage],
    Generic[UplinkPackage, DownlinkPackage, DiskSharedData],
):
    def __init__(self, num_parallels: int, share_dir: Path) -> None:
        self.num_parallels = num_parallels
        self.share_dir = share_dir
        self.share_dir.mkdir(parents=True, exist_ok=True)
        self.cache: list[UplinkPackage] = []

    @abstractmethod
    def get_shared_data(self, cid: int, payload: DownlinkPackage) -> DiskSharedData: ...

    @staticmethod
    @abstractmethod
    def process_client(path: Path) -> Path: ...

    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None:
        pool = mp.Pool(processes=self.num_parallels)
        jobs: list[ApplyResult] = []

        for cid in cid_list:
            path = self.share_dir.joinpath(f"{cid}.pkl")
            data = self.get_shared_data(cid, payload)
            torch.save(data, path)
            jobs.append(pool.apply_async(self.process_client, (path,)))

        for job in tqdm(jobs, desc="Client", leave=False):
            path = job.get()
            assert isinstance(path, Path)
            package = torch.load(path, weights_only=False)
            self.cache.append(package)

        pool.close()
        pool.join()
