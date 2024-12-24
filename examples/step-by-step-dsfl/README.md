# Step-by-Step Tutorial: DS-FL

Welcome to this step-by-step tutorial on implementing [DS-FL](https://doi.ieeecomputersociety.org/10.1109/TMC.2021.3070013) using BlazeFL. DS-FL is a Federated Learning method that performs knowledge distillation by exchanging model outputs on an open dataset. 

Thanks to its high extensibility, the BlazeFL framework allows for the simple implementation of not only typical parameter exchange methods but also distillation-based methods and other innovative approaches, much like assembling puzzle pieces. 

This tutorial will surely assist you in implementing your unique and unconventional FL methods, helping you break free from traditional constraints.

## Setup a Project

First, open your terminal and create a new directory for your project.

```bash
mkdir step-by-step-dsfl
cd step-by-step-dsfl
```

Next, Initialize the projcet with [uv](https://github.com/astral-sh/uv) or your favorite package manager.

```bash
uv init
```

Then, create a virtual environment and install BlazeFL. 

```bash
uv venv --python 3.12
source .venv/bin/activate
uv add blazefl
```

## Implementing a PartitionedDataset

By pre-splitting and saving the dataset in advance, the server or clients can retrieve data directly without needing to repartition it for every round.
In BlazeFL, it is recommended to implement your dataset by extending the abstract class `PartitionedDataset`.

For example, in DS-FL, you can implement the `DSFLPartitionedDataset` class as follows:

```python
from blazefl.core import PartitionedDataset

class DSFLPartitionedDataset(PartitionedDataset):
    # Omited for brevity

    def get_dataset(self, type_: str, cid: int | None) -> Dataset:
        match type_:
            case "train" | "val":
                dataset = torch.load(
                    self.path.joinpath(type_, f"{cid}.pkl"),
                    weights_only=False,
                )
            case "open":
                dataset = torch.load(
                    self.path.joinpath(type_, "open.pkl"),
                    weights_only=False,
                )
            case "test":
                dataset = torch.load(
                    self.path.joinpath(type_, "test.pkl"), weights_only=False
                )
            case _:
                raise ValueError("Invalid type_")
        assert isinstance(dataset, Dataset)
        return dataset

    def get_dataloader(
        self, type_: str, cid: int | None, batch_size: int | None = None
    ) -> DataLoader:
        dataset = self.get_dataset(type_, cid)
        assert isinstance(dataset, Sized)
        batch_size = len(dataset) if batch_size is None else batch_size
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return data_loader
```

The `PartitionedDataset` class requires the implementation of two methods: `get_dataset` and `get_dataloader`.
The `get_dataset` method returns a `Dataset` based on the specified type and client ID, while the `get_dataloader` method provides the corresponding `DataLoader`.
This design allows for flexibility, making it suitable even for unconventional approaches like DS-FL, which utilizes an open dataset.
If only one of these methods is needed, or if neither is required, you can simply use pass in their implementation.

The complete source code can be found [here](https://github.com/kitsuya0828/BlazeFL/tree/main/examples/step-by-step-dsfl/dataset).

## Implementing a ModelSelector

Traditional FL frameworks generally assume that all clients use the same model.
However, in distillation-based FL methods like DS-FL, it is possible for clients to use different models.

To accommodate this, BlazeFL provides an abstract class called `ModelSelector`, which allows the server and clients to dynamically select models as needed.

For example, in DS-FL, you can implement the `DSFLModelSelector` class as follows:

```python
from blazefl.core import ModelSelector

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
                raise ValueError
```

The ModelSelector class requires the implementation of the `select_model` method.
This method is straightforward: it takes a model name as input and returns the corresponding `nn.Module`.
Variables such as the number of classes in a classification task can be stored as attributes of the `ModelSelector` class for convenience.

The complete source code can be found [here](https://github.com/kitsuya0828/BlazeFL/tree/main/examples/step-by-step-dsfl/models).


## Defining DownlinkPackage and UplinkPackage

In traditional FL frameworks, data exchanged between the server and clients is typically represented using generic data structures like lists or dictionaries.
However, this approach lacks type definitions for individual objects within the package, making debugging challenging and reducing code readability.

BlazeFL addresses this issue by recommending the use of `DownlinkPackage` and `UplinkPackage` for defining types for data exchanged between the server and clients.
By leveraging type inference, debugging becomes easier within the editor, and the overall code readability improves significantly.

For instance, in DS-FL, you can define the `DSFLDownlinkPackage` and `DSFLUplinkPackage` as follows:

```python
@dataclass
class DSFLUplinkPackage:
    soft_labels: torch.Tensor
    indices: torch.Tensor
    metadata: dict


@dataclass
class DSFLDownlinkPackage:
    soft_labels: torch.Tensor | None
    indices: torch.Tensor | None
    next_indices: torch.Tensor
```

Using the `@dataclass` decorator allows for concise and readable class definitions.
This approach not only simplifies the implementation but also provides a clear structure for the data being exchanged, making it easier to maintain and extend.


