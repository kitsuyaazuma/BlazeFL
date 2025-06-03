from abc import ABC, abstractmethod
from typing import Generic, TypeVar

UplinkPackage = TypeVar('UplinkPackage')
DownlinkPackage = TypeVar('DownlinkPackage')

class ServerHandler(ABC, Generic[UplinkPackage, DownlinkPackage]):
    @abstractmethod
    def downlink_package(self) -> DownlinkPackage: ...
    @abstractmethod
    def sample_clients(self) -> list[int]: ...
    @abstractmethod
    def if_stop(self) -> bool: ...
    @abstractmethod
    def global_update(self, buffer: list[UplinkPackage]) -> None: ...
    @abstractmethod
    def load(self, payload: UplinkPackage) -> bool: ...
