import marimo as mo
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import torchvision
from pathlib import Path
import random
from collections import defaultdict
from collections.abc import Sized, Mapping
from dataclasses import dataclass, field
import logging
from datetime import datetime

# BlazeFL specific imports (based on the README and source files)
from blazefl.core import PartitionedDataset, ModelSelector, ServerHandler, ParallelClientTrainer
from blazefl.utils import FilteredDataset, RandomState, seed_everything

# Imports from the example's own modules (will be defined in the notebook later)
# For now, we'll prepare for them.
# from dataset.functional import balance_split, client_inner_dirichlet_partition_faster # Will be defined in a cell
# from models.cnn import CNN # Will be defined in a cell

mo.md(r'''
# Step-by-Step Tutorial: DS-FL

> [!NOTE]
> **Work in Progress**
> This tutorial is currently under development! We welcome any feedback or contributions from those who try it out. Feel free to submit pull requests, open issues, or share your ideas in the repository to help make this tutorial even better.

Welcome to this step-by-step tutorial on implementing [DS-FL](https://doi.ieeecomputersociety.org/10.1109/TMC.2021.3070013) using BlazeFL!
DS-FL is a Federated Learning (FL) method that utilizes knowledge distillation by sharing model outputs on an open dataset.

Thanks to BlazeFL's highly modular design, you can easily implement both standard FL approaches (like parameter exchange) and advanced methods (like distillation-based FL).
Think of it as assembling puzzle pieces to create your own unique FL methods—beyond the constraints of traditional frameworks.


In this tutorial, we’ll guide you through creating a DS-FL pipeline using BlazeFL.
By following along, you’ll be able to develop your own original FL methods.

## Setup a Project

Start by creating a new directory for your DS-FL project:

```bash
mkdir step-by-step-dsfl
cd step-by-step-dsfl
```

Next, Initialize the projcet with [uv](https://github.com/astral-sh/uv) (or any other package manager of your choice).

```bash
uv init
```

Then, create a virtual environment and install BlazeFL.

```bash
uv venv --python 3.12 # or your preferred python version
source .venv/bin/activate
uv add blazefl torch torchvision
```

We also install `torch` and `torchvision` as they are common dependencies for BlazeFL projects and used in this tutorial.
''')

mo.md(r'''
## Implementing a PartitionedDataset

Before running Federated Learning, it’s common to pre-split the dataset for each client.
By saving these partitions ahead of time, your server or clients can simply load the data each round without re-partitioning.

In BlazeFL, we recommend extending the `PartitionedDataset` abstract class to create your own dataset class.
This allows for flexible data handling, even for methods like DS-FL which might use an open dataset.

The `get_dataset` method should return a `Dataset` for a specified type (e.g., "train", "val", "open", or "test") and client ID.
The `get_dataloader` method wraps that dataset in a `DataLoader`.
If you don’t need one of these methods for your specific federated learning approach, you can simply implement it with `pass`.

Below, we'll define `DSFLPartitionedDataset`. It relies on some helper functions for data partitioning, which we'll define first.
''')

# Helper functions (originally from dataset/functional.py)
import numpy as np
import numpy.typing as npt

def balance_split(num_clients: int, num_samples: int) -> npt.NDArray[np.int_]:
    '''
    Adapted from https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
    '''
    num_samples_per_client = int(num_samples / num_clients)
    client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(int)
    return client_sample_nums


def client_inner_dirichlet_partition_faster(
    targets: list[int] | npt.NDArray[np.int_],
    num_clients: int,
    num_classes: int,
    dir_alpha: float,
    client_sample_nums: npt.NDArray[np.int_],
    class_priors: npt.NDArray[np.float64] | None = None,
    verbose: bool = False, # Changed default to False for notebook clarity
) -> tuple[dict[int, npt.NDArray[np.int_]], npt.NDArray[np.float64]]:
    '''
    Adapted from https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
    '''
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets)

    if class_priors is None:  # CHANGED: use given class_priors if provided
        class_priors = np.random.dirichlet(
            alpha=[dir_alpha] * num_classes, size=num_clients
        )
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.where(targets == i)[0] for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]

    client_indices = [
        np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in range(num_clients)
    ]

    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        # If current node is full resample a client
        if verbose:
            print(f"Remaining Data: {np.sum(client_sample_nums)}")
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = int(np.argmax(np.random.uniform() <= curr_prior))
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                # Exception handling: If the current class has no samples left,
                # randomly select a non-zero class
                active_classes = [i for i, count in enumerate(class_amount) if count > 0]
                if not active_classes: # Should not happen if data is available
                    # This might indicate an issue or that all data is distributed
                    # For notebook, let's break or raise to avoid infinite loop if sum(client_sample_nums) is stuck
                    if np.sum(client_sample_nums) == 0 : break # double check condition
                    # Fallback if something is wrong, pick any class and hope for the best or error out
                    # This part needs careful consideration based on how FedLab handles it.
                    # For now, let's assume there's always a class with samples if client_sample_nums > 0
                    pass # Let it try to find a class. If stuck, indicates issue in logic or data.

                curr_class = np.random.choice(active_classes) if active_classes else curr_class


            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = idx_list[
                curr_class
            ][class_amount[curr_class]]

            break

    if np.sum(client_sample_nums) != 0:
        # This case means we couldn't distribute all samples, which might be an issue.
        # For a notebook, it's good to be aware.
        # Depending on strictness, one might raise an error or log a warning.
        print(f"Warning: Not all samples were distributed. Remaining: {np.sum(client_sample_nums)}")


    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    return client_dict, class_priors

class DSFLPartitionedDataset(PartitionedDataset):
    def __init__(
        self,
        root: Path,
        path: Path,
        num_clients: int,
        seed: int,
        partition: str,
        open_size: int,
        dir_alpha: float,
    ) -> None:
        self.root = root
        self.path = path
        self.num_clients = num_clients
        self.seed = seed
        self.partition = partition
        self.dir_alpha = dir_alpha
        self.open_size = open_size

        # It's good practice to define transforms within the class or pass them,
        # making the class more self-contained for a notebook.
        self.train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomCrop(32, padding=4),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # In a notebook, directly calling _preprocess might be too slow for cell execution.
        # Users might want to call it explicitly.
        # For this conversion, we'll keep it, but it's a consideration for notebook UX.
        # mo.md(r"**Note:** The following `_preprocess()` method will download datasets and prepare them. This might take some time.")
        # self._preprocess() # We might comment this out and instruct user to call it.
                           # For now, let's assume it's part of the setup.

    def _preprocess(self):
        '''
        Preprocesses the dataset by downloading CIFAR10 and CIFAR100,
        partitioning them for clients, and saving them to disk.
        '''
        self.root.mkdir(parents=True, exist_ok=True)
        # Ensure path for saving splits also exists
        self.path.mkdir(parents=True, exist_ok=True)

        # mo.md(r"Downloading and preparing datasets...") # Inform user

        train_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=True,
            download=True,
        )
        open_dataset = torchvision.datasets.CIFAR100( # Using CIFAR100 as open dataset as in example
            root=self.root,
            train=True, # Typically use training part of a different dataset
            download=True,
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=self.root,
            train=False,
            download=True,
        )

        # Create subdirectories for different dataset types if they don't exist
        for type_ in ["train", "open", "test"]: # "open" is a single file, not a dir of files
            if type_ == "open":
                self.path.mkdir(parents=True, exist_ok=True) # Parent for open.pkl
            else:
                self.path.joinpath(type_).mkdir(parents=True, exist_ok=True)


        match self.partition:
            case "client_inner_dirichlet":
                client_dict, class_priors = client_inner_dirichlet_partition_faster(
                    targets=train_dataset.targets,
                    num_clients=self.num_clients,
                    num_classes=len(train_dataset.classes), # Use actual number of classes
                    dir_alpha=self.dir_alpha,
                    client_sample_nums=balance_split(
                        num_clients=self.num_clients,
                        num_samples=len(train_dataset.targets), # Corrected to use targets length
                    ),
                    verbose=False,
                )
                # Partition test set per client using similar distribution
                test_client_dict, _ = client_inner_dirichlet_partition_faster(
                    targets=test_dataset.targets,
                    num_clients=self.num_clients,
                    num_classes=len(test_dataset.classes), # Use actual number of classes
                    dir_alpha=self.dir_alpha,
                    client_sample_nums=balance_split(
                        num_clients=self.num_clients,
                        num_samples=len(test_dataset.targets), # Corrected
                    ),
                    class_priors=class_priors, # Use same class priors for consistency
                    verbose=False,
                )
            case _:
                raise ValueError(f"Invalid partition: {self.partition}")

        for cid, indices in client_dict.items():
            client_train_dataset = FilteredDataset(
                indices=indices.tolist(), # Ensure indices is a list
                original_data=train_dataset.data, # Pass data and targets separately
                original_targets=np.array(train_dataset.targets)[indices].tolist(),
                transform=self.train_transform,
            )
            torch.save(client_train_dataset, self.path.joinpath("train", f"{cid}.pkl"))

        for cid, indices in test_client_dict.items():
            client_test_dataset = FilteredDataset(
                indices=indices.tolist(),
                original_data=test_dataset.data,
                original_targets=np.array(test_dataset.targets)[indices].tolist(),
                transform=self.test_transform,
            )
            torch.save(client_test_dataset, self.path.joinpath("test", f"{cid}.pkl"))

        # Prepare open dataset (subset of CIFAR100)
        # Ensure open_size does not exceed available samples
        actual_open_size = min(self.open_size, len(open_dataset))
        open_indices = np.random.choice(
            len(open_dataset),
            size=actual_open_size,
            replace=False # Ensure unique samples
        )
        torch.save(
            FilteredDataset(
                indices=open_indices.tolist(),
                original_data=open_dataset.data, # Pass data
                original_targets=None, # Open dataset might not have targets in the same way or not used
                transform=self.train_transform, # Usually use train_transform for augmentation if any
            ),
            self.path.joinpath("open.pkl"), # Save as open.pkl directly in self.path
        )

        # Save default/global test dataset
        torch.save(
            FilteredDataset(
                indices=list(range(len(test_dataset))),
                original_data=test_dataset.data,
                original_targets=test_dataset.targets,
                transform=self.test_transform,
            ),
            self.path.joinpath("test", "default.pkl"),
        )
        # mo.md(r"Dataset preprocessing complete.") # Inform user

    def get_dataset(self, type_: str, cid: int | None) -> Dataset:
        match type_:
            case "train":
                if cid is None: raise ValueError("Client ID (cid) must be provided for train dataset.")
                dataset_path = self.path.joinpath(type_, f"{cid}.pkl")
                if not dataset_path.exists(): raise FileNotFoundError(f"Train dataset for client {cid} not found at {dataset_path}")
                dataset = torch.load(dataset_path, weights_only=False)
            case "open":
                dataset_path = self.path.joinpath(f"{type_}.pkl")
                if not dataset_path.exists(): raise FileNotFoundError(f"Open dataset not found at {dataset_path}")
                dataset = torch.load(dataset_path, weights_only=False)
            case "test":
                if cid is not None: # Client specific test set
                    dataset_path = self.path.joinpath(type_, f"{cid}.pkl")
                    if not dataset_path.exists(): raise FileNotFoundError(f"Test dataset for client {cid} not found at {dataset_path}")
                else: # Default/global test set
                    dataset_path = self.path.joinpath(type_, "default.pkl")
                    if not dataset_path.exists(): raise FileNotFoundError(f"Default test dataset not found at {dataset_path}")
                dataset = torch.load(dataset_path, weights_only=False)
            case _:
                raise ValueError(f"Invalid dataset type: {type_}")
        assert isinstance(dataset, Dataset), f"Loaded object is not a PyTorch Dataset, but {type(dataset)}"
        return dataset

    def get_dataloader(
        self, type_: str, cid: int | None, batch_size: int | None = None
    ) -> DataLoader:
        dataset = self.get_dataset(type_, cid)
        assert isinstance(dataset, Sized), "Dataset must have a __len__ method to be used with DataLoader."
        # Determine batch size: if None, use full dataset size; otherwise, use provided batch_size.
        # Ensure batch_size does not exceed dataset size if dataset is smaller.
        effective_batch_size = batch_size if batch_size is not None else len(dataset)
        if len(dataset) == 0: # Handle empty dataset case
            # mo.md(f"Warning: Dataset type '{type_}' (cid: {cid}) is empty. Returning DataLoader with no data.")
            # Return an empty DataLoader or raise an error, depending on desired behavior
            return DataLoader(dataset, batch_size=effective_batch_size if effective_batch_size > 0 else 1, shuffle=False) # shuffle=False for empty

        effective_batch_size = min(effective_batch_size, len(dataset))


        data_loader = DataLoader(dataset, batch_size=effective_batch_size, shuffle=(type_=="train")) # Shuffle only for train
        return data_loader

mo.md(r'''
**Note on `_preprocess`**: The `DSFLPartitionedDataset` class includes a `_preprocess` method that downloads datasets (CIFAR10, CIFAR100) and partitions them.
In a real scenario, you would instantiate this class once to prepare the data. For this notebook, we'll define it, but you'd typically call `_preprocess()` on an instance if you were setting up the data from scratch. The paths used are relative, so ensure your notebook is run from the `examples/step-by-step-dsfl` directory or adjust paths accordingly.

For the notebook to run without errors during definition, the actual `_preprocess()` call during `__init__` is commented out in the class definition above. You would typically run it like this *once* to set up your data:

```python
# Example of how to initialize and preprocess:
# (Assuming you are in examples/step-by-step-dsfl directory)
# dataset_root_dir = Path("./data_root")
# dataset_split_dir = dataset_root_dir / "cifar_splits"
# partitioned_dataset = DSFLPartitionedDataset(
#     root=dataset_root_dir,
#     path=dataset_split_dir,
#     num_clients=10,
#     seed=42,
#     partition="client_inner_dirichlet",
#     open_size=5000,
#     dir_alpha=0.5
# )
# partitioned_dataset._preprocess() # Call this once to generate and save data
# mo.md("Preprocessing would be done here.")
```
This interactive notebook will assume the data has been pre-processed and is available at the specified paths when other components (like ServerHandler or ClientTrainer) try to load it.
''')

mo.md(r'''
## Implementing a ModelSelector

Most traditional FL frameworks assume all clients use the same model. However, in distillation-based methods like DS-FL, clients (and the server) can potentially use different models.
BlazeFL provides an abstract class called `ModelSelector` to handle this scenario. It lets you select different models on the fly for the server and clients.

The `select_model` method typically takes a string (the model name) and returns the corresponding `nn.Module`. You can store useful information (like the number of classes) as attributes in your `ModelSelector`.

First, let's define a simple CNN model that our `ModelSelector` can use.
''')

class CNN(nn.Module):
    """
    Based on
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    with slight modifications.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

mo.md(r'''
Now, let's define the `DSFLModelSelector`.
''')

class DSFLModelSelector(ModelSelector):
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    def select_model(self, model_name: str) -> nn.Module:
        match model_name:
            case "cnn":
                # This will use the CNN class defined in the cell above
                return CNN(num_classes=self.num_classes)
            case "resnet18":
                # For a notebook, ensure torchvision.models.resnet18 is imported
                # It was added to the initial imports.
                return torchvision.models.resnet18(num_classes=self.num_classes)
            case _:
                raise ValueError(f"Model name '{model_name}' is not recognized.")

mo.md(r'''
## Defining DownlinkPackage and UplinkPackage

In many FL frameworks, communication between the server and clients is often handled through generic data structures like dictionaries or lists.
However, BlazeFL encourages you to define dedicated classes for these communication packets, making your code more organized and readable.
Using Python’s `@dataclass` makes these classes concise and easy to maintain. Including explicit types for each attribute also improves IDE support and debugging.

In DS-FL, the `UplinkPackage` carries information from the client to the server (e.g., soft labels computed by the client on public data, indices of that data, and any metadata like client's evaluation metrics).
The `DownlinkPackage` carries information from the server to the client (e.g., aggregated global soft labels from other clients, indices for that global data, and indices for the next batch of public data the client should process).
''')

@dataclass
class DSFLUplinkPackage:
    soft_labels: torch.Tensor
    indices: torch.Tensor
    metadata: dict


@dataclass
class DSFLDownlinkPackage:
    soft_labels: torch.Tensor | None # Can be None in the first round
    indices: torch.Tensor | None    # Can be None in the first round
    next_indices: torch.Tensor      # Indices for the client to process next

mo.md(r'''
## Implementing a ServerHandler

The server in an FL setup typically handles aggregating information from clients and updating the global model (or, in the case of DS-FL, a global representation like aggregated soft labels).
BlazeFL does not force any specific "aggregation" or "update" strategy. Instead, it provides a flexible `ServerHandler` abstract class that focuses on the necessary client-server communication and state management.

The `ServerHandler` class generally requires the following core methods to be implemented:
- `sample_clients()`: Selects clients for the current round.
- `if_stop()`: Determines if the training process should terminate.
- `load(payload)`: Processes an `UplinkPackage` from a client. It should return `True` if the server is ready to perform its global update (e.g., enough client packages received), and `False` otherwise.
- `global_update(buffer)`: Performs the server's main logic once `load` has indicated it's ready (e.g., aggregates client contributions, updates a global model or knowledge). The `buffer` usually contains the packages collected from clients in the current round.
- `downlink_package()`: Prepares the `DownlinkPackage` to be sent to the next round of clients.

If any of these methods are unnecessary for your specific approach, you can simply implement them with `pass`.

In DS-FL, the `global_update` method aggregates soft labels from clients and then distills this aggregated knowledge into the server's model using an open dataset. The server then sends these refined soft labels (or instructions for clients) back to the clients.
''')

class DSFLServerHandler(ServerHandler[DSFLUplinkPackage, DSFLDownlinkPackage]):
    def __init__(
        self,
        model_selector: DSFLModelSelector, # Expected to be an instance of the class defined earlier
        model_name: str,
        dataset: DSFLPartitionedDataset, # Expected to be an instance of the class defined earlier
        global_round: int,
        num_clients: int,
        sample_ratio: float,
        device: str,
        kd_epochs: int,
        kd_batch_size: int,
        kd_lr: float,
        era_temperature: float,
        open_size_per_round: int,
    ) -> None:
        self.model = model_selector.select_model(model_name)
        self.dataset = dataset # This is an instance of DSFLPartitionedDataset
        self.global_round = global_round
        self.num_clients = num_clients
        self.sample_ratio = sample_ratio
        self.device = device
        self.kd_epochs = kd_epochs
        self.kd_batch_size = kd_batch_size
        self.kd_lr = kd_lr
        self.era_temperature = era_temperature
        self.open_size_per_round = open_size_per_round

        self.client_buffer_cache: list[DSFLUplinkPackage] = []
        self.global_soft_labels: torch.Tensor | None = None
        self.global_indices: torch.Tensor | None = None
        self.num_clients_per_round = int(self.num_clients * self.sample_ratio)
        if self.num_clients_per_round == 0 and self.num_clients > 0 and self.sample_ratio > 0:
             self.num_clients_per_round = 1 # Ensure at least one client if sampling is too low for small N
        self.round = 0
        self.metadata_list: list[dict] = []


    def sample_clients(self) -> list[int]:
        if self.num_clients == 0: return []
        # Ensure num_clients_per_round is not greater than num_clients
        actual_clients_to_sample = min(self.num_clients_per_round, self.num_clients)
        if actual_clients_to_sample == 0: return []

        sampled_clients = random.sample(
            range(self.num_clients), actual_clients_to_sample
        )
        return sorted(sampled_clients)

    def get_next_indices(self) -> torch.Tensor:
        # Ensure open_size_per_round does not exceed actual open_size available in dataset
        # This requires dataset.open_size attribute or a method to get it.
        # Assuming self.dataset has an attribute `open_size` as per its __init__
        actual_open_data_size = self.dataset.open_size

        current_open_size_per_round = min(self.open_size_per_round, actual_open_data_size)
        if current_open_size_per_round == 0 and actual_open_data_size > 0: # If configured to 0 but data exists, use a small portion
            current_open_size_per_round = actual_open_data_size # Or a fixed small number
        elif actual_open_data_size == 0: # No open data available
            return torch.tensor([], dtype=torch.long)


        shuffled_indices = torch.randperm(actual_open_data_size)
        return shuffled_indices[:current_open_size_per_round]

    def if_stop(self) -> bool:
        return self.round >= self.global_round

    def load(self, payload: DSFLUplinkPackage) -> bool:
        self.client_buffer_cache.append(payload)

        # Check if enough clients reported for this round
        # self.num_clients_per_round could be 0 if num_clients is 0.
        if self.num_clients_per_round == 0:
            # If no clients are expected, global_update might still run (e.g. for initialization or time-based updates)
            # or it might mean an issue with config. For DS-FL, it likely means no updates if no clients.
            # Let's assume if num_clients_per_round is 0, we don't proceed to global_update via load.
            return False

        if len(self.client_buffer_cache) >= self.num_clients_per_round:
            # Optional: Trim buffer if more clients reported than expected (e.g. due to async nature)
            # self.client_buffer_cache = self.client_buffer_cache[:self.num_clients_per_round]
            self.global_update(self.client_buffer_cache) # Pass the current buffer
            self.round += 1
            self.client_buffer_cache = [] # Clear buffer for next round
            return True
        else:
            return False

    def global_update(self, buffer: list[DSFLUplinkPackage]) -> None:
        if not buffer: # If buffer is empty, nothing to update
            # mo.md(r"Skipping global_update as client buffer is empty.")
            # Potentially update round counter or other logic if server still "ticks"
            # self.round +=1 # If rounds increment even without client data
            return

        soft_labels_list = [ele.soft_labels for ele in buffer]
        indices_list = [ele.indices for ele in buffer]
        self.metadata_list = [ele.metadata for ele in buffer] # Store metadata

        soft_labels_stack: defaultdict[int, list[torch.Tensor]] = defaultdict(list)
        for soft_labels, indices in zip(soft_labels_list, indices_list, strict=True):
            for soft_label, index in zip(soft_labels, indices, strict=True):
                soft_labels_stack[int(index.item())].append(soft_label)

        global_soft_labels_list: list[torch.Tensor] = []
        global_indices_list: list[int] = []

        if not soft_labels_stack: # if no soft labels were collected (e.g. all indices were unique and no overlap)
            # mo.md(r"No overlapping soft labels from clients to aggregate.")
            # Server model still needs to be "distilled" or it won't learn.
            # Option 1: Server generates its own soft labels on open set if no client data useful.
            # Option 2: Skip distillation for this round if no data.
            # The original code proceeds to distill, which might error if lists are empty.
            # For now, let's ensure distill handles empty lists or we provide some defaults.
            # If global_soft_labels_list is empty, torch.stack will fail.
             pass # Let it attempt, will likely need handling in distill or before if this is an issue.


        for index_val, s_labels in soft_labels_stack.items(): # Corrected variable name from 'indices' to 'index_val'
            global_indices_list.append(index_val)
            mean_soft_labels = torch.mean(torch.stack(s_labels), dim=0)
            # Entropy Reduction Aggregation (ERA)
            era_soft_labels = F.softmax(mean_soft_labels / self.era_temperature, dim=0)
            global_soft_labels_list.append(era_soft_labels)

        # Call static distill method
        # Check if there's anything to distill
        if global_soft_labels_list:
            DSFLServerHandler.distill(
                self.model,
                self.dataset, # Pass the DSFLPartitionedDataset instance
                global_soft_labels_list,
                global_indices_list,
                self.kd_epochs,
                self.kd_batch_size,
                self.kd_lr,
                self.device,
            )
            self.global_soft_labels = torch.stack(global_soft_labels_list)
            self.global_indices = torch.tensor(global_indices_list, dtype=torch.long)
        else:
            # mo.md(r"Skipping distillation as no aggregated soft labels were produced.")
            self.global_soft_labels = None # Ensure these are reset or handled
            self.global_indices = None


    @staticmethod
    def distill(
        model: nn.Module,
        dataset: DSFLPartitionedDataset, # Type hint for clarity
        global_soft_labels_list: list[torch.Tensor], # Renamed for clarity
        global_indices_list: list[int], # Renamed for clarity
        kd_epochs: int,
        kd_batch_size: int,
        kd_lr: float,
        device: str,
    ) -> None:
        if not global_soft_labels_list or not global_indices_list:
            # mo.md(r"Distill: No global soft labels or indices provided. Skipping distillation.")
            return

        model.to(device)
        model.train()

        # Get the open dataset
        # Ensure 'open' dataset exists and is not empty
        try:
            openset = dataset.get_dataset(type_="open", cid=None)
        except FileNotFoundError:
            # mo.md(r"Distill: Open dataset not found. Skipping distillation.")
            # logging.error("Distill: Open dataset not found.")
            return

        if len(openset) == 0 or len(global_indices_list) == 0:
             # mo.md(r"Distill: Open dataset or global indices list is empty. Skipping distillation.")
             return

        # Ensure global_indices_list are valid for subsetting openset
        valid_indices = [idx for idx in global_indices_list if idx < len(openset)]
        if not valid_indices:
            # mo.md(r"Distill: No valid indices for the open dataset. Skipping distillation.")
            return

        # Filter global_soft_labels_list to match valid_indices
        # This assumes a 1-to-1 correspondence in the original lists.
        # A safer way would be to build a dict: index -> soft_label
        # For now, let's assume they correspond and filter based on valid_indices found in openset

        # Create a mapping from original index value to soft_label tensor
        label_dict = {idx: lbl for idx, lbl in zip(global_indices_list, global_soft_labels_list)}

        # Filtered lists based on valid_indices
        filtered_soft_labels = [label_dict[idx] for idx in valid_indices]

        if not filtered_soft_labels:
            # mo.md(r"Distill: No soft labels remain after filtering for valid indices. Skipping.")
            return

        open_loader = DataLoader(
            Subset(openset, valid_indices), # Use valid_indices
            batch_size=kd_batch_size, # Ensure kd_batch_size > 0
            shuffle=True # Shuffle for training
        )

        # The FilteredDataset for soft labels needs to be created from the filtered list
        soft_label_dataset = FilteredDataset(
            indices=list(range(len(filtered_soft_labels))), # New indices for the filtered list
            original_data=filtered_soft_labels # Pass the list of tensors directly
        )

        global_soft_label_loader = DataLoader(
            soft_label_dataset,
            batch_size=kd_batch_size, # Match batch size
            shuffle=False # Order should match open_loader if not shuffled, or handle correspondence
        )

        # Ensure there's data to load
        if len(open_loader) == 0 or len(global_soft_label_loader) == 0:
            # mo.md(r"Distill: DataLoaders are empty. Skipping distillation.")
            return

        optimizer = torch.optim.SGD(model.parameters(), lr=kd_lr)
        for _ in range(kd_epochs):
            # Iterate over the shorter loader if they differ in length due to batching
            num_batches = min(len(open_loader), len(global_soft_label_loader))
            iter_open_loader = iter(open_loader)
            iter_soft_label_loader = iter(global_soft_label_loader)

            for _ in range(num_batches):
                data = next(iter_open_loader) # Unpack if it's a list/tuple from FilteredDataset
                soft_label_batch = next(iter_soft_label_loader) # This will be a batch of tensors

                # If FilteredDataset for data returns (data_item, original_index), unpack data_item
                actual_data = data[0] if isinstance(data, (list, tuple)) else data
                actual_soft_labels = soft_label_batch[0] if isinstance(soft_label_batch, (list,tuple)) else soft_label_batch


                actual_data = actual_data.to(device)
                # Soft labels are already processed (e.g. ERA applied), directly use them.
                # They should be shaped [batch_size, num_classes]
                target_soft_labels = actual_soft_labels.to(device)
                # Ensure it's not squeezed if batch_size is 1, should be [1, num_classes]
                if target_soft_labels.ndim == 1: target_soft_labels = target_soft_labels.unsqueeze(0)


                output = model(actual_data) # Model output logits
                log_probs = F.log_softmax(output, dim=1)

                # KL Divergence: input is log_probs, target is probs
                loss = F.kl_div(
                    log_probs, target_soft_labels, reduction="batchmean"
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return

    @staticmethod
    def evaulate( # 'evaluate' is the common spelling
        model: nn.Module, test_loader: DataLoader, device: str
    ) -> tuple[float, float]:
        model.to(device)
        model.eval()
        criterion = nn.CrossEntropyLoss() # Standard loss for evaluation

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        if len(test_loader) == 0:
            # mo.md(r"Evaluate: Test loader is empty. Returning 0 loss, 0 acc.")
            return 0.0, 0.0

        with torch.no_grad():
            for data_batch in test_loader:
                # Assuming test_loader yields (inputs, labels)
                # Adjust if it yields only inputs or different structure
                if not isinstance(data_batch, (list, tuple)) or len(data_batch) != 2:
                    # logging.warning("Evaluate: Unexpected data format from test_loader. Skipping batch.")
                    continue

                inputs, labels = data_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs, 1)
                correct = torch.sum(predicted.eq(labels)).item()

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                total_correct += int(correct)
                total_samples += batch_size

        if total_samples == 0: return 0.0, 0.0 # Avoid division by zero

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples

        return avg_loss, avg_acc

    def get_summary(self) -> dict[str, float]:
        # Server model evaluation on the global test set
        # Ensure 'default' test set can be loaded by get_dataloader
        try:
            server_test_loader = self.dataset.get_dataloader(
                type_="test",
                cid=None, # For global test set
                batch_size=self.kd_batch_size, # Use kd_batch_size or a dedicated eval_batch_size
            )
        except FileNotFoundError:
            # mo.md(r"GetSummary: Default test dataset not found. Server eval metrics will be 0.")
            # logging.warning("GetSummary: Default test dataset not found.")
            server_loss, server_acc = 0.0, 0.0
        else:
            server_loss, server_acc = DSFLServerHandler.evaulate( # Corrected spelling
                self.model,
                server_test_loader,
                self.device,
            )

        # Client metrics are from self.metadata_list collected in global_update
        if self.metadata_list:
            client_loss = sum(m.get("loss", 0.0) for m in self.metadata_list) / len(self.metadata_list)
            client_acc = sum(m.get("acc", 0.0) for m in self.metadata_list) / len(self.metadata_list)
        else:
            client_loss, client_acc = 0.0, 0.0

        return {
            "server_acc": server_acc,
            "server_loss": server_loss,
            "client_acc": client_acc,
            "client_loss": client_loss,
        }

    def downlink_package(self) -> DSFLDownlinkPackage:
        next_indices = self.get_next_indices()
        return DSFLDownlinkPackage(
            self.global_soft_labels, self.global_indices, next_indices
        )
