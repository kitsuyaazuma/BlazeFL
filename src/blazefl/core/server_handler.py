from abc import ABC, abstractmethod
from typing import Any


class ServerHandler(ABC):
    @abstractmethod
    def downlink_package(self) -> Any: ...

    @abstractmethod
    def sample_clients(self) -> list[int]: ...

    @abstractmethod
    def if_stop(self) -> bool: ...

    @abstractmethod
    def global_update(self, buffer: list[Any]) -> None: ...

    @abstractmethod
    def load(self, payload: Any) -> bool: ...
