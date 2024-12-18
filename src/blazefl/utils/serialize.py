import torch


def serialize_model(model: torch.nn.Module, cpu: bool = True) -> torch.Tensor:
    """
    Serialize a PyTorch model's parameters into a flat tensor.

    Args:
        model (torch.nn.Module): The PyTorch model to serialize.
        cpu (bool): Whether to move the serialized parameters to the CPU.

    Returns:
        torch.Tensor: A flat tensor containing the serialized parameters.
    """
    parameters = [param.data.view(-1) for param in model.state_dict().values()]
    serialized_parameters = torch.cat(parameters)
    if cpu:
        serialized_parameters = serialized_parameters.cpu()

    return serialized_parameters


def deserialize_model(
    model: torch.nn.Module, serialized_parameters: torch.Tensor
) -> None:
    """
    Deserialize a flat tensor back into a PyTorch model's parameters.

    Args:
        model (torch.nn.Module): The PyTorch model to update.
        serialized_parameters (torch.Tensor): The tensor containing the parameters.

    Returns:
        None
    """
    current_index = 0
    for param in model.state_dict().values():
        numel = param.numel()
        size = param.size()
        param.copy_(
            serialized_parameters[current_index : current_index + numel].view(size)
        )
        current_index += numel
