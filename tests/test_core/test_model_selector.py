import pytest
import torch

from src.blazefl.core import ModelSelector


class DummyModelSelector(ModelSelector):
    def select_model(self, model_name: str) -> torch.nn.Module:
        match model_name:
            case "linear":
                return torch.nn.Linear(10, 5)
            case "conv":
                return torch.nn.Conv2d(1, 3, 3)
            case _:
                raise ValueError("Unknown model")


def test_model_selector_subclass() -> None:
    selector = DummyModelSelector()
    model = selector.select_model("linear")
    assert isinstance(model, torch.nn.Linear)
    model = selector.select_model("conv")
    assert isinstance(model, torch.nn.Conv2d)

    with pytest.raises(ValueError):
        selector.select_model("unknown")
