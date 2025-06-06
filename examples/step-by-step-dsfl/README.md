# Step-by-Step Tutorial: DS-FL

Welcome to this step-by-step tutorial on implementing DS-FL[^1] using BlazeFL!
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

Next, Initialize the project with [uv](https://github.com/astral-sh/uv) (or any other package manager of your choice).

```bash
uv init --python 3.12
```

Then, create a virtual environment and install BlazeFL. 

```bash
uv venv
# source .venv/bin/activate
uv add blazefl
```

## Implementing a PartitionedDataset

Before running Federated Learning, it’s common to pre-split the dataset for each client.
By saving these partitions ahead of time, your server or clients can simply load the data each round without re-partitioning.

In BlazeFL, we recommend extending the `PartitionedDataset` abstract class to create your own dataset class. 
For example, you can implement `DSFLPartitionedDataset` like this:

```python
from blazefl.core import PartitionedDataset

class DSFLPartitionedDataset(PartitionedDataset):
    # Omited for brevity

    def get_dataset(self, type_: str, cid: int | None) -> Dataset:
        match type_:
            case "train":
                dataset = torch.load(
                    self.path.joinpath(type_, f"{cid}.pkl"),
                    weights_only=False,
                )
            case "open":
                dataset = torch.load(
                    self.path.joinpath(f"{type_}.pkl"),
                    weights_only=False,
                )
            case "test":
                if cid is not None:
                    dataset = torch.load(
                        self.path.joinpath(type_, f"{cid}.pkl"),
                        weights_only=False,
                    )
                else:
                    dataset = torch.load(
                        self.path.joinpath(type_, "default.pkl"), weights_only=False
                    )
            case _:
                raise ValueError(f"Invalid dataset type: {type_}")
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

Here, `get_dataset` returns a `Dataset` for the specified type (e.g., "train", "open", or "test") and client ID.
Meanwhile, `get_dataloader` wraps that dataset in a `DataLoader`.
This design is flexible enough even for methods like DS-FL, which rely on an open dataset.
If you don’t need one of these methods, you can simply implement it with `pass`.

You can view the complete source code [here](https://github.com/kitsuyaazuma/blazefl/tree/main/examples/step-by-step-dsfl/dataset).

## Implementing a ModelSelector

Most traditional FL frameworks assume all clients use the same model, but in distillation-based methods like DS-FL, clients can use different models.

BlazeFL provides an abstract class called `ModelSelector` to handle this scenario.
It lets you select different models on the fly for the server and clients.
For instance:

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
                raise ValueError(f"Invalid model name: {model_name}")
```

Here, `select_model` simply takes a string (the model name) and returns the corresponding `nn.Module`.
You can store useful information (like the number of classes) as attributes in your `ModelSelector`.

The full source code can be found [here](https://github.com/kitsuyaazuma/blazefl/tree/main/examples/step-by-step-dsfl/models).

## Defining DownlinkPackage and UplinkPackage

In many FL frameworks, communication between the server and clients is often handled through generic data structures like dictionaries or lists.
However, BlazeFL encourages you to define dedicated classes for these communication packets, making your code more organized and readable.

In DS-FL, you could define them like this:

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

Using Python’s `@dataclass` makes these classes concise and easy to maintain.
Including explicit types for each attribute also improves IDE support for debugging.

## Implementing a ServerHandler

The server in an FL setup typically handles aggregating information from clients and updating the global model.
BlazeFL does not force any specific "aggregation" or "update" strategy.
Instead, it provides a flexible `ServerHandler` class that focuses on the necessary client-server communication.

Below is an example for DS-FL:

```python
class DSFLServerHandler(ServerHandler[DSFLUplinkPackage, DSFLDownlinkPackage]):
    # Omitted for brevity

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
            self.kd_optimizer,
            self.dataset,
            global_soft_labels,
            global_indices,
            self.kd_epochs,
            self.kd_batch_size,
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

The `ServerHandler` class requires five core methods to be implemented:

- `sample_clients`
- `if_stop`
- `load`
- `global_update`
- `downlink_package`

If any of these methods are not needed for your approach, you can simply implement them with `pass`.

In DS-FL, the `global_update` method aggregates the soft labels from clients and distills them into a global model.
However, you have the flexibility to place any custom operations in these or other methods.
You can find more details in the [official documentation](https://kitsuyaazuma.github.io/blazefl/generated/blazefl.core.ServerHandler.html#blazefl.core.ServerHandler).


## Implementing a ParallelClientTrainer

Traditional FL frameworks often train each client sequentially and upload parameters to the server.
With BlazeFL, the `ParallelClientTrainer` class lets you train multiple clients in parallel while retaining full extensibility.

An example DS-FL client trainer looks like this:

```python
@dataclass
class DSFLDiskSharedData:
    # Omitted for brevity

class DSFLParallelClientTrainer(
    ParallelClientTrainer[DSFLUplinkPackage, DSFLDownlinkPackage, DSFLDiskSharedData]
):
    # Omitted for brevity

    @staticmethod
    def process_client(path: Path) -> Path:
        data = torch.load(path, weights_only=False)
        assert isinstance(data, DSFLDiskSharedData)

        model = data.model_selector.select_model(data.model_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=data.lr)
        kd_optimizer: torch.optim.SGD | None = None

        state: DSFLClientState | None = None
        if data.state_path.exists():
            state = torch.load(data.state_path, weights_only=False)
            assert isinstance(state, DSFLClientState)
            RandomState.set_random_state(state.random)
            model.load_state_dict(state.model)
            optimizer.load_state_dict(state.optimizer)
            if state.kd_optimizer is not None:
                kd_optimizer = torch.optim.SGD(model.parameters(), lr=data.kd_lr)
                kd_optimizer.load_state_dict(state.kd_optimizer)
        else:
            seed_everything(data.seed, device=device)

        # Distill
        open_dataset = data.dataset.get_dataset(type_="open", cid=None)
        if data.payload.indices is not None and data.payload.soft_labels is not None:
            global_soft_labels = list(torch.unbind(data.payload.soft_labels, dim=0))
            global_indices = data.payload.indices.tolist()
            if kd_optimizer is None:
                kd_optimizer = torch.optim.SGD(model.parameters(), lr=data.kd_lr)
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
            Subset(open_dataset, data.payload.next_indices.tolist()),
            batch_size=data.batch_size,
        )
        soft_labels = DSFLParallelClientTrainer.predict(
            model=model,
            open_loader=open_loader,
            device=data.device,
        )

        # Evaluate
        test_loader = data.dataset.get_dataloader(
            type_="val",
            cid=data.cid,
            batch_size=data.batch_size,
        )
        loss, acc = DSFLServerHandler.evaluate(
            model=model,
            test_loader=test_loader,
            device=data.device,
        )

        package = DSFLUplinkPackage(
            soft_labels=soft_labels,
            indices=data.payload.next_indices,
            metadata={"loss": loss, "acc": acc},
        )

        torch.save(package, path)
        state = DSFLClientState(
            random=RandomState.get_random_state(device=data.device),
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            kd_optimizer=kd_optimizer.state_dict() if kd_optimizer else None,
        )
        torch.save(state, data.state_path)
        return path

    def get_shared_data(
        self, cid: int, payload: DSFLDownlinkPackage
    ) -> DSFLDiskSharedData:
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

This class uses Python’s standard library multiprocessing (wrapped under BlazeFL) to train clients concurrently.
You mainly need to implement:

- `process_client` (a static method called by child processes)
- `get_shared_data` (to prepare the data shared across processes)
- `uplink_package` (to send final results back to the server)

By storing shared data on disk instead of passing it directly, you avoid complex shared memory management.
This design makes it straightforward to enable parallel training.

The complete source code is [here](https://github.com/kitsuyaazuma/blazefl/tree/main/examples/step-by-step-dsfl/algorithm/dsfl.py).

## Implementing a Pipeline

A `Pipeline` is optional but can help organize your simulation workflow, making it easy to run experiments in a structured way.
Here’s an example DS-FL pipeline:

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

This pipeline is almost identical to one you might create for FedAvg or another standard FL method, showcasing how reusable these components are. 

In this snippet, we use TensorBoard via SummaryWriter for logging, but you’re free to use alternatives like [W&B](https://github.com/wandb/wandb).

You can see the full source code [here](https://github.com/kitsuyaazuma/BlazeFL/tree/main/examples/step-by-step-dsfl/main.py).

## Running the Simulation

In our example, we use [Hydra](https://hydra.cc/) to handle hyperparameter configuration. Feel free to use any configuration system you like.

To run the DS-FL simulation:

```bash
uv run python main.py +algorithm=dsfl
```

To visualize metrics in TensorBoard:

```bash
make visualize
```

## Conclusion

In this tutorial, you learned how to implement DS-FL using BlazeFL.
BlazeFL’s flexible design eliminates many constraints seen in traditional FL frameworks, allowing you to mix and match components like building blocks.

Use BlazeFL to implement your own original FL methods and drive pioneering research in Federated Learning.
Push boundaries and have fun exploring innovative approaches!

[^1]: S. Itahara, T. Nishio, Y. Koda, M. Morikura, and K. Yamamoto, "Distillation-Based Semi-Supervised Federated Learning for Communication-Efficient Collaborative Training With Non-IID Private Data," IEEE Trans. Mobile Comput., vol. 22, no. 1, pp. 191–205, 2023.
