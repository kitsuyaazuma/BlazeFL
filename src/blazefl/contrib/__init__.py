"""
Federated Learning Algorithms Implementations.

This module provides implementations of various federated learning algorithms,
extending the core functionalities of BlazeFL.
"""

from blazefl.contrib.fedavg import (
    FedAvgParalleClientTrainer,
    FedAvgSerialClientTrainer,
    FedAvgServerHandler,
)

__all__ = [
    "FedAvgServerHandler",
    "FedAvgParalleClientTrainer",
    "FedAvgSerialClientTrainer",
]
