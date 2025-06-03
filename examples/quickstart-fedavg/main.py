import logging
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.multiprocessing as mp
from dataset import PartitionedCIFAR10
from hydra.core import hydra_config
from models import FedAvgModelSelector
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms

from blazefl.contrib import (
    FedAvgParallelClientTrainer,
    FedAvgSerialClientTrainer,
    FedAvgServerHandler,
)
from blazefl.utils import seed_everything


class FedAvgPipeline:
    def __init__(
        self,
        handler: FedAvgServerHandler,
        trainer: FedAvgSerialClientTrainer | FedAvgParallelClientTrainer,
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

            metadata_list = [
                pack.metadata for pack in uploads if pack.metadata is not None
            ]
            avg_loss = sum(meta["loss"] for meta in metadata_list) / len(metadata_list)
            avg_acc = sum(meta["acc"] for meta in metadata_list) / len(metadata_list)

            logging.info(
                f"Round: {round_}, Loss: {avg_loss:.2f}, Accuracy: {avg_acc:.2f}"
            )
            self.writer.add_scalar("Loss", avg_loss, round_)
            self.writer.add_scalar("Accuracy", avg_acc, round_)

            # server side
            for pack in uploads:
                self.handler.load(pack)

        logging.info("Done!")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(
    cfg: DictConfig,
):
    print(OmegaConf.to_yaml(cfg))

    log_dir = hydra_config.HydraConfig.get().runtime.output_dir
    writer = SummaryWriter(log_dir=log_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_root_dir = Path(cfg.dataset_root_dir)
    dataset_split_dir = dataset_root_dir.joinpath(timestamp)
    share_dir = Path(cfg.share_dir).joinpath(timestamp)
    state_dir = Path(cfg.state_dir).joinpath(timestamp)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(cfg.seed, device=device)

    dataset = PartitionedCIFAR10(
        root=dataset_root_dir,
        path=dataset_split_dir,
        num_clients=cfg.num_clients,
        num_shards=cfg.num_shards,
        dir_alpha=cfg.dir_alpha,
        seed=cfg.seed,
        partition=cfg.partition,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        ),
    )
    model_selector = FedAvgModelSelector(num_classes=10)
    handler = FedAvgServerHandler(
        model_selector=model_selector,
        model_name=cfg.model_name,
        dataset=dataset,
        global_round=cfg.global_round,
        num_clients=cfg.num_clients,
        device=device,
        sample_ratio=cfg.sample_ratio,
    )
    if cfg.serial:
        trainer = FedAvgSerialClientTrainer(
            model_selector=model_selector,
            model_name=cfg.model_name,
            dataset=dataset,
            device=device,
            num_clients=cfg.num_clients,
            epochs=cfg.epochs,
            lr=cfg.lr,
            batch_size=cfg.batch_size,
        )
    else:
        trainer = FedAvgParallelClientTrainer(
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
        )
    pipeline = FedAvgPipeline(handler=handler, trainer=trainer, writer=writer)
    try:
        pipeline.main()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")
    except Exception as e:
        logging.error(e)


if __name__ == "__main__":
    # NOTE: To use CUDA with multiprocessing, you must use the 'spawn' start method
    mp.set_start_method("spawn")

    main()
