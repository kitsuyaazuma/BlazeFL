from blazefl.core import ModelSelector
from torch import nn
from torchvision.models import resnet18

from models.cnn import CNN


class DSFLModelSelector(ModelSelector):
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def select_model(self, model_name: str) -> nn.Module:
        match model_name:
            case "cnn":
                return CNN(num_classes=self.num_classes)
            case "resnet18":
                return resnet18(num_classes=self.num_classes)
            case _:
                raise ValueError(f"Invalid model name: {model_name}")
