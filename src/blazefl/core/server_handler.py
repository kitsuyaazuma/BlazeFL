from abc import ABC, abstractmethod
from typing import Any


class ServerHandler(ABC):
    """
    Abstract base class for server-side operations in federated learning.

    This class defines the essential methods that a server handler must implement
    to manage communication and coordination with clients during federated learning
    processes.

    Raises:
        NotImplementedError: If any of the methods are not implemented in a subclass.
    """

    @abstractmethod
    def downlink_package(self) -> Any:
        """
        Prepare the data package to be sent from the server to clients.

        Returns:
            Any: The data package intended for client consumption.
        """
        ...

    @abstractmethod
    def sample_clients(self) -> list[int]:
        """
        Select a list of client IDs to participate in the current training round.

        Returns:
            list[int]: A list of selected client IDs.
        """
        ...

    @abstractmethod
    def if_stop(self) -> bool:
        """
        Determine whether the federated learning process should be terminated.

        Returns:
            bool: True if the process should stop; False otherwise.
        """
        ...

    @abstractmethod
    def global_update(self, buffer: list[Any]) -> None:
        """
        Update the global model based on the aggregated data from clients.

        Args:
            buffer (list[Any]): A list containing data from clients to be aggregated.

        Returns:
            None
        """
        ...

    @abstractmethod
    def load(self, payload: Any) -> bool:
        """
        Load a given payload into the server's state.

        Args:
            payload (Any): The data to be loaded into the server.

        Returns:
            bool: True if the payload was successfully loaded; False otherwise.
        """
        ...
