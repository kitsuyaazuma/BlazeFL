import logging
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from blazefl.contrib import (
    FedAvgBaseClientTrainer,
    FedAvgBaseServerHandler,
    FedAvgProcessPoolClientTrainer,
)
from blazefl.contrib.fedavg import FedAvgDownlinkPackage, FedAvgUplinkPackage
from blazefl.core import ModelSelector, PartitionedDataset, ThreadPoolClientTrainer
from blazefl.utils import seed_everything
from omegaconf import DictConfig, OmegaConf

from dataset import PartitionedCIFAR10
from models import FedAvgModelSelector


class FedAvgMultiThreadClientTrainer(
    ThreadPoolClientTrainer[
        FedAvgUplinkPackage,
        FedAvgDownlinkPackage,
    ]
):
    def __init__(
        self,
        model_selector: ModelSelector,
        model_name: str,
        dataset: PartitionedDataset,
        device: str,
        num_clients: int,
        epochs: int,
        batch_size: int,
        lr: float,
        seed: int,
        num_parallels: int,
    ) -> None:
        self.num_parallels = num_parallels
        self.device = device
        if self.device == "cuda":
            self.device_count = torch.cuda.device_count()
        self.cache: list[FedAvgUplinkPackage] = []

        self.model_selector = model_selector
        self.model_name = model_name
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_clients = num_clients
        self.seed = seed

    def worker(
        self,
        cid: int,
        device: str,
        payload: FedAvgDownlinkPackage,
    ) -> FedAvgUplinkPackage:
        model = self.model_selector.select_model(self.model_name)
        train_loader = self.dataset.get_dataloader(
            type_="train",
            cid=cid,
            batch_size=self.batch_size,
        )
        package = FedAvgProcessPoolClientTrainer.train(
            model=model,
            model_parameters=payload.model_parameters,
            train_loader=train_loader,
            device=device,
            epochs=self.epochs,
            lr=self.lr,
        )
        return package

    def uplink_package(self) -> list[FedAvgUplinkPackage]:
        package = deepcopy(self.cache)
        self.cache = []
        return package


class FedAvgPipeline:
    def __init__(
        self,
        handler: FedAvgBaseServerHandler,
        trainer: FedAvgBaseClientTrainer
        | FedAvgProcessPoolClientTrainer
        | FedAvgMultiThreadClientTrainer,
    ) -> None:
        self.handler = handler
        self.trainer = trainer

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
            formatted_summary = ", ".join(f"{k}: {v:.3f}" for k, v in summary.items())
            logging.info(f"round: {round_}, {formatted_summary}")

        logging.info("done!")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root_dir = Path(cfg.dataset_root_dir)
    dataset_split_dir = dataset_root_dir.joinpath(timestamp)
    share_dir = Path(cfg.share_dir).joinpath(timestamp)
    state_dir = Path(cfg.state_dir).joinpath(timestamp)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    logging.info(f"device: {device}")

    seed_everything(cfg.seed, device=device)

    dataset = PartitionedCIFAR10(
        root=dataset_root_dir,
        path=dataset_split_dir,
        num_clients=cfg.num_clients,
        seed=cfg.seed,
        partition=cfg.partition,
        num_shards=cfg.num_shards,
        dir_alpha=cfg.dir_alpha,
    )
    model_selector = FedAvgModelSelector(num_classes=10)

    handler = FedAvgBaseServerHandler(
        model_selector=model_selector,
        model_name=cfg.model_name,
        dataset=dataset,
        global_round=cfg.global_round,
        num_clients=cfg.num_clients,
        device=device,
        sample_ratio=cfg.sample_ratio,
        batch_size=cfg.batch_size,
    )
    trainer: (
        FedAvgBaseClientTrainer
        | FedAvgProcessPoolClientTrainer
        | FedAvgMultiThreadClientTrainer
        | None
    ) = None
    match cfg.execution_mode:
        case "serial":
            trainer = FedAvgBaseClientTrainer(
                model_selector=model_selector,
                model_name=cfg.model_name,
                dataset=dataset,
                device=device,
                num_clients=cfg.num_clients,
                epochs=cfg.epochs,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
            )
        case "multi-process":
            trainer = FedAvgProcessPoolClientTrainer(
                model_selector=model_selector,
                model_name=cfg.model_name,
                dataset=dataset,
                share_dir=share_dir,
                state_dir=state_dir,
                seed=cfg.seed,
                device=device,
                num_clients=cfg.num_clients,
                epochs=cfg.epochs,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
                num_parallels=cfg.num_parallels,
                ipc_mode=cfg.ipc_mode,
            )
        case "multi-thread":
            trainer = FedAvgMultiThreadClientTrainer(
                model_selector=model_selector,
                model_name=cfg.model_name,
                dataset=dataset,
                device=device,
                num_clients=cfg.num_clients,
                epochs=cfg.epochs,
                lr=cfg.lr,
                batch_size=cfg.batch_size,
                num_parallels=cfg.num_parallels,
                seed=cfg.seed,
            )
        case _:
            raise ValueError(f"Invalid execution mode: {cfg.execution_mode}")
    pipeline = FedAvgPipeline(handler=handler, trainer=trainer)
    try:
        pipeline.main()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt: Stopping the pipeline.")
    except Exception as e:
        logging.exception(f"An error occurred: {e}")


if __name__ == "__main__":
    # NOTE: To use CUDA with multiprocessing, you must use the 'spawn' start method
    mp.set_start_method("spawn")

    main()
