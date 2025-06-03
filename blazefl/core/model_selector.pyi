import torch
from abc import ABC, abstractmethod

class ModelSelector(ABC):
    @abstractmethod
    def select_model(self, model_name: str) -> torch.nn.Module: ...
