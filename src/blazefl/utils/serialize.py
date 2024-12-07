import torch


def serialize_model(model: torch.nn.Module) -> torch.Tensor:
    parameters = [param.data.view(-1) for param in model.state_dict().values()]
    serialized_parameters = torch.cat(parameters)

    return serialized_parameters


def deserialize_model(
    model: torch.nn.Module, serialized_parameters: torch.Tensor
) -> None:
    current_index = 0
    for param in model.state_dict().values():
        numel = param.numel()
        size = param.size()
        param.copy_(
            serialized_parameters[current_index : current_index + numel].view(size)
        )
        current_index += numel
