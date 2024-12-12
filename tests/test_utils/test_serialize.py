import pytest
import torch

from src.blazefl.utils import deserialize_model, serialize_model


@pytest.fixture
def simple_model() -> torch.nn.Module:
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 3), torch.nn.ReLU(), torch.nn.Linear(3, 1)
    )
    return model


def test_serialize_deserialize_cpu(simple_model: torch.nn.Module) -> None:
    original_params = [p.clone() for p in simple_model.parameters()]
    serialized = serialize_model(simple_model, cpu=True)

    assert serialized.device.type == "cpu"

    total_numel = sum(p.numel() for p in simple_model.parameters())
    assert serialized.numel() == total_numel

    for p in simple_model.parameters():
        p.data.normal_()

    deserialize_model(simple_model, serialized)

    for orig_p, new_p in zip(original_params, simple_model.parameters(), strict=True):
        assert torch.allclose(orig_p, new_p), "Parameters did not restore correctly."


def test_serialize_deserialize_gpu(simple_model: torch.nn.Module) -> None:
    if torch.cuda.is_available():
        simple_model = simple_model.to("cuda")
        original_params = [p.clone() for p in simple_model.parameters()]

        serialized_gpu = serialize_model(simple_model, cpu=False)
        assert serialized_gpu.device.type == "cuda"

        serialized_cpu = serialize_model(simple_model, cpu=True)
        assert serialized_cpu.device.type == "cpu"

        for p in simple_model.parameters():
            p.data.normal_()

        simple_model = simple_model.cpu()
        deserialize_model(simple_model, serialized_cpu)

        for orig_p, new_p in zip(
            original_params, simple_model.parameters(), strict=True
        ):
            assert torch.allclose(
                orig_p.cpu(), new_p
            ), "Parameters did not restore correctly."
