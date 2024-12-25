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

## Implementing a ServerHandler

In traditional FL, the server is responsible for tasks like aggregating data and updating models.
These tasks can vary significantly depending on the use case.
BlazeFL enhances flexibility by not imposing any predefined aggregation or update functions.
Instead, it provides the `ServerHandler` class, which focuses solely on client-server communication as a mandatory requirement.

For instance, in DS-FL, the `DSFLServerHandler` class can be implemented by extending the `ServerHandler` class:

```python
class DSFLServerHandler(ServerHandler[DSFLUplinkPackage, DSFLDownlinkPackage]):
    # Omited for brevity

    def sample_clients(self) -> list[int]:
        sampled_clients = random.sample(
            range(self.num_clients), self.num_clients_per_round
        )

        return sorted(sampled_clients)

    def if_stop(self) -> bool:
        return self.round >= self.global_round

    def load(self, payload: DSFLUplinkPackage) -> bool:
        self.client_buffer_cache.append(payload)

        if len(self.client_buffer_cache) == self.num_clients_per_round:
            self.global_update(self.client_buffer_cache)
            self.round += 1
            self.client_buffer_cache = []
            return True
        else:
            return False

    def global_update(self, buffer) -> None:
        soft_labels_list = [ele.soft_labels for ele in buffer]
        indices_list = [ele.indices for ele in buffer]
        self.metadata_list = [ele.metadata for ele in buffer]

        soft_labels_stack: defaultdict[int, list[torch.Tensor]] = defaultdict(
            list[torch.Tensor]
        )
        for soft_labels, indices in zip(soft_labels_list, indices_list, strict=True):
            for soft_label, index in zip(soft_labels, indices, strict=True):
                soft_labels_stack[int(index.item())].append(soft_label)

        global_soft_labels: list[torch.Tensor] = []
        global_indices: list[int] = []
        for indices, soft_labels in soft_labels_stack.items():
            global_indices.append(indices)
            mean_soft_labels = torch.mean(torch.stack(soft_labels), dim=0)
            # Entropy Reduction Aggregation (ERA)
            era_soft_labels = F.softmax(mean_soft_labels / self.era_temperature, dim=0)
            global_soft_labels.append(era_soft_labels)

        DSFLServerHandler.distill(
            self.model,
            self.dataset,
            global_soft_labels,
            global_indices,
            self.kd_epochs,
            self.kd_batch_size,
            self.kd_lr,
            self.device,
        )

        self.global_soft_labels = torch.stack(global_soft_labels)
        self.global_indices = torch.tensor(global_indices)
    
    # Omited for brevity

    def downlink_package(self) -> DSFLDownlinkPackage:
        next_indices = self.get_next_indices()
        return DSFLDownlinkPackage(
            self.global_soft_labels, self.global_indices, next_indices
        )
```

The `ServerHandler` class requires the implementation of five key methods: `sample_clients`, `if_stop`, `load, global_update`, and `downlink_package`.
You can find detailed explanations of each method in the [official documentation](https://kitsuya0828.github.io/BlazeFL/generated/blazefl.core.ServerHandler.html#blazefl.core.ServerHandler)
If any of these methods are unnecessary for your use case, they can simply be implemented with pass.

In the case of DS-FL, the `global_update` method aggregates the soft labels received from clients and performs model distillation. However, BlazeFL allows you to implement any custom logic in any desired order, ensuring high flexibility to meet your requirements.

## Implementing a ParallelClientTrainer

In traditional FL frameworks, clients are typically trained sequentially, and their model parameters are uploaded to the server.
However, BlazeFL provides the `ParallelClientTrainer` class, which enables parallel training of clients while maintaining high extensibility.

For example, in DS-FL, the `DSFLParallelClientTrainer` class can be implemented as follows:

```python
@dataclass
class DSFLDiskSharedData:
    # Omited for brevity

class DSFLParallelClientTrainer(
    ParallelClientTrainer[DSFLUplinkPackage, DSFLDownlinkPackage, DSFLDiskSharedData]
):
    # Omited for brevity

    @staticmethod
    def process_client(path: Path) -> Path:
        data = torch.load(path, weights_only=False)
        assert isinstance(data, DSFLDiskSharedData)

        if data.state_path.exists():
            state = torch.load(data.state_path, weights_only=False)
            assert isinstance(state, RandomState)
            RandomState.set_random_state(state)
        else:
            seed_everything(data.seed, device=data.device)

        model = data.model_selector.select_model(data.model_name)

        # Distill
        openset = data.dataset.get_dataset(type_="open", cid=None)
        if data.payload.indices is not None and data.payload.soft_labels is not None:
            global_soft_labels = list(torch.unbind(data.payload.soft_labels, dim=0))
            global_indices = data.payload.indices.tolist()
            DSFLServerHandler.distill(
                model=model,
                dataset=data.dataset,
                global_soft_labels=global_soft_labels,
                global_indices=global_indices,
                kd_epochs=data.kd_epochs,
                kd_batch_size=data.kd_batch_size,
                kd_lr=data.kd_lr,
                device=data.device,
            )

        # Train
        train_loader = data.dataset.get_dataloader(
            type_="train",
            cid=data.cid,
            batch_size=data.batch_size,
        )
        DSFLParallelClientTrainer.train(
            model=model,
            train_loader=train_loader,
            device=data.device,
            epochs=data.epochs,
            lr=data.lr,
        )

        # Predict
        open_loader = DataLoader(
            Subset(openset, data.payload.next_indices.tolist()),
            batch_size=data.batch_size,
        )
        soft_labels = DSFLParallelClientTrainer.predict(
            model=model,
            open_loader=open_loader,
            device=data.device,
        )

        # Evaluate
        val_loader = data.dataset.get_dataloader(
            type_="val",
            cid=data.cid,
            batch_size=data.batch_size,
        )
        loss, acc = DSFLServerHandler.evaulate(
            model=model,
            test_loader=val_loader,
            device=data.device,
        )

        package = DSFLUplinkPackage(
            soft_labels=torch.stack(soft_labels),
            indices=data.payload.next_indices,
            metadata={"loss": loss, "acc": acc},
        )

        torch.save(package, path)
        torch.save(RandomState.get_random_state(device=data.device), data.state_path)
        return path

    def get_shared_data(
        self, cid: int, payload: DSFLDownlinkPackage
    ) -> DSFLDiskSharedData:
        if self.device == "cuda":
            device = f"cuda:{cid % self.device_count}"
        else:
            device = self.device
        data = DSFLDiskSharedData(
            model_selector=self.model_selector,
            model_name=self.model_name,
            dataset=self.dataset,
            epochs=self.epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            kd_epochs=self.kd_epochs,
            kd_batch_size=self.kd_batch_size,
            kd_lr=self.kd_lr,
            device=device,
            cid=cid,
            seed=self.seed,
            payload=payload,
            state_path=self.state_dir.joinpath(f"{cid}.pt"),
        )
        return data

    def uplink_package(self) -> list[DSFLUplinkPackage]:
        package = deepcopy(self.cache)
        self.cache: list[DSFLUplinkPackage] = []
        return package
```


The `ParallelClientTrainer` class utilizes [multiprocessing](https://docs.python.org/ja/3/library/multiprocessing.html) to enable parallelization.
While it hides the complexity of `multiprocessing` through its pre-implemented `local_process` method, this method relies solely on Python’s standard library, making it understandable and customizable if needed.

The methods that need to be implemented are `uplink_package`, `get_shared_data`, and `process_client`.
Among these, `process_client` is a static method executed by child processes, while `get_shared_data` returns data shared among clients.
This shared data is stored on disk, and only the path to it is passed to child processes via shared memory.
This approach avoids the complexity of shared memory management while enabling efficient and simple parallelization.

The complete source code can be found [here](https://github.com/kitsuya0828/BlazeFL/tree/main/examples/step-by-step-dsfl/algorithm/dsfl.py).


## Implementing a Pipeline

Although optional, combining these components into a `Pipeline` allows for flexible execution of simulations. 

For instance, a pipeline for DS-FL can be implemented as follows:

```python
class DSFLPipeline:
    def __init__(
        self,
        handler: DSFLServerHandler,
        trainer: DSFLParallelClientTrainer,
        writer: SummaryWriter,
    ) -> None:
        self.handler = handler
        self.trainer = trainer
        self.writer = writer

    def main(self):
        while not self.handler.if_stop():
            round_ = self.handler.round
            # server side
            sampled_clients = self.handler.sample_clients()
            broadcast = self.handler.downlink_package()

            # client side
            self.trainer.local_process(broadcast, sampled_clients)
            uploads = self.trainer.uplink_package()

            # server side
            for pack in uploads:
                self.handler.load(pack)

            summary = self.handler.get_summary()
            for key, value in summary.items():
                self.writer.add_scalar(key, value, round_)
            logging.info(f"Round {round_}: {summary}")

        logging.info("Done!")
```

This implementation is nearly identical to a pipeline for FedAvg, demonstrating its reusability across multiple methods.
In this example, metrics are saved and visualized using TensorBoard, but other tools like [W&B](https://github.com/wandb/wandb) can also be used.

By initializing the ServerHandler and ParallelClientTrainer and passing them into the pipeline, you can execute the simulation.

The full source code is available [here](https://github.com/kitsuya0828/BlazeFL/tree/main/examples/step-by-step-dsfl/main.py).

## Running the Simulation
In this example, [Hydra](https://hydra.cc/) is used to configure hyperparameters, but you can choose your own method for configuration.

Run the DS-FL simulation with the following command:

```bash
uv run python main.py +algorithm=dsfl
```

You can also visualize the metrics using TensorBoard:

```bash
make visualize
```

## Conclusion


In this tutorial, we demonstrated how to implement DS-FL using BlazeFL.
BlazeFL offers a level of flexibility not found in traditional FL frameworks, enabling you to implement your unique FL methods by combining components like puzzle pieces.

Take advantage of BlazeFL to implement your original FL methods and push the boundaries of innovative research.

