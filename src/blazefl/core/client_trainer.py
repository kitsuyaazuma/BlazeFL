from abc import ABC, abstractmethod
from typing import Callable, TypeVar, Any


class SerialClientTrainer(ABC):
    @abstractmethod
    def uplink_package(self) -> list[Any]: ...

    @abstractmethod
    def local_process(self, payload: Any, cid_list: list[int]) -> None: ...


M = TypeVar("M")  # Shared Memory
D = TypeVar("D")  # Disk


class ParallelClientTrainer(ABC):
    @abstractmethod
    def uplink_package(self) -> list[Any]: ...

    # @abstractmethod
    def get_client_worker(self, M, D) -> Callable[[M, D], tuple[M, D]]: ...

    @abstractmethod
    def local_process(self, payload: Any, cid_list: list[int]) -> None: ...
