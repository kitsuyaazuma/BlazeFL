from multiprocessing.pool import ApplyResult as ApplyResult
from pathlib import Path
from typing import Protocol, TypeVar

UplinkPackage = TypeVar('UplinkPackage')
DownlinkPackage = TypeVar('DownlinkPackage', contravariant=True)

class SerialClientTrainer(Protocol[UplinkPackage, DownlinkPackage]):
    def uplink_package(self) -> list[UplinkPackage]: ...
    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None: ...
DiskSharedData = TypeVar('DiskSharedData', covariant=True)

class ParallelClientTrainer(Protocol[UplinkPackage, DownlinkPackage, DiskSharedData]):
    num_parallels: int
    share_dir: Path
    device: str
    device_count: int
    cache: list[UplinkPackage]
    def uplink_package(self) -> list[UplinkPackage]: ...
    def get_shared_data(self, cid: int, payload: DownlinkPackage) -> DiskSharedData: ...
    def get_client_device(self, cid: int) -> str: ...
    @staticmethod
    def process_client(path: Path, device: str) -> Path: ...
    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None: ...
