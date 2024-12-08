from abc import ABC, abstractmethod

import torch


class ModelSelector(ABC):
    @abstractmethod
    def select_model(self, model_name: str) -> torch.nn.Module: ...
