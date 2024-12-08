from abc import ABC, abstractmethod
from typing import Any


class SerialClientTrainer(ABC):
    @abstractmethod
    def uplink_package(self) -> list[Any]: ...

    @abstractmethod
    def local_process(self, payload: Any, cid_list: list[int]) -> None: ...


class ParallelClientTrainer(ABC):
    @abstractmethod
    def uplink_package(self) -> list[Any]: ...

    @abstractmethod
    def local_process(self, payload: Any, cid_list: list[int]) -> None: ...
