"""
Federated Learning Algorithms Implementations.

This module provides implementations of various federated learning algorithms,
extending the core functionalities of BlazeFL.
"""

from blazefl.contrib.fedavg import (
    FedAvgBaseClientTrainer,
    FedAvgProcessPoolClientTrainer,
    FedAvgServerHandler,
)

__all__ = [
    "FedAvgServerHandler",
    "FedAvgProcessPoolClientTrainer",
    "FedAvgBaseClientTrainer",
]
