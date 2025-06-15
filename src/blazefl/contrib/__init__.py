"""
Federated Learning Algorithms Implementations.

This module provides implementations of various federated learning algorithms,
extending the core functionalities of BlazeFL.
"""

from blazefl.contrib.fedavg import (
    FedAvgBaseClientTrainer,
    FedAvgBaseServerHandler,
    FedAvgDownlinkPackage,
    FedAvgProcessPoolClientTrainer,
    FedAvgThreadPoolClientTrainer,
    FedAvgUplinkPackage,
)

__all__ = [
    "FedAvgBaseServerHandler",
    "FedAvgProcessPoolClientTrainer",
    "FedAvgBaseClientTrainer",
    "FedAvgThreadPoolClientTrainer",
    "FedAvgUplinkPackage",
    "FedAvgDownlinkPackage",
]
