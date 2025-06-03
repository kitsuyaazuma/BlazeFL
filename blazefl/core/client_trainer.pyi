from _typeshed import Incomplete
from abc import ABC, abstractmethod
from multiprocessing.pool import ApplyResult as ApplyResult
from pathlib import Path
from typing import Generic, TypeVar

UplinkPackage = TypeVar('UplinkPackage')
DownlinkPackage = TypeVar('DownlinkPackage')

class SerialClientTrainer(ABC, Generic[UplinkPackage, DownlinkPackage]):
    @abstractmethod
    def uplink_package(self) -> list[UplinkPackage]: ...
    @abstractmethod
    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None: ...
DiskSharedData = TypeVar('DiskSharedData')

class ParallelClientTrainer(SerialClientTrainer[UplinkPackage, DownlinkPackage], Generic[UplinkPackage, DownlinkPackage, DiskSharedData]):
    num_parallels: Incomplete
    share_dir: Incomplete
    device: Incomplete
    device_count: Incomplete
    cache: Incomplete
    def __init__(self, num_parallels: int, share_dir: Path, device: str) -> None: ...
    @abstractmethod
    def get_shared_data(self, cid: int, payload: DownlinkPackage) -> DiskSharedData: ...
    def get_client_device(self, cid: int) -> str: ...
    @staticmethod
    @abstractmethod
    def process_client(path: Path, device: str) -> Path: ...
    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None: ...
