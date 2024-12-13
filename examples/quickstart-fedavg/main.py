import argparse
import logging
from datetime import datetime
from pathlib import Path

import torch
import torch.multiprocessing as mp
from dataset import PartitionedCIFAR10
from models import FedAvgModelSelector
from torchvision import transforms

from blazefl.contrib import (
    FedAvgParalleClientTrainer,
    FedAvgSerialClientTrainer,
    FedAvgServerHandler,
)


class FedAvgPipeline:
    def __init__(
        self,
        handler: FedAvgServerHandler,
        trainer: FedAvgSerialClientTrainer | FedAvgParalleClientTrainer,
    ) -> None:
        self.handler = handler
        self.trainer = trainer

    def main(self):
        while not self.handler.if_stop():
            logging.info(f"[ROUND {self.handler.round}]")
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
                f"Average Loss: {avg_loss:.2f}, " f"Average Accuracy: {avg_acc:.2f}"
            )

            # server side
            for pack in uploads:
                self.handler.load(pack)

        logging.info("[FINISHED]")


def main(
    model_name: str,
    num_clients: int,
    global_round: int,
    sample_ratio: float,
    partition: str,
    num_shards: int,
    dir_alpha: float,
    seed: int,
    epochs: int,
    lr: float,
    batch_size: int,
    num_parallels: int,
    dataset_root_dir: Path,
    dataset_split_dir: Path,
    share_dir: Path,
    state_dir: Path,
    device: str,
    serial: bool,
):
    dataset = PartitionedCIFAR10(
        root=dataset_root_dir,
        path=dataset_split_dir,
        num_clients=num_clients,
        num_shards=num_shards,
        dir_alpha=dir_alpha,
        seed=seed,
        partition=partition,
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
        model_name=model_name,
        dataset=dataset,
        global_round=global_round,
        num_clients=num_clients,
        device=device,
        sample_ratio=sample_ratio,
    )
    if serial:
        trainer = FedAvgSerialClientTrainer(
            model_selector=model_selector,
            model_name=model_name,
            dataset=dataset,
            device=device,
            num_clients=num_clients,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
        )
    else:
        trainer = FedAvgParalleClientTrainer(
            model_selector=model_selector,
            model_name=model_name,
            dataset=dataset,
            share_dir=share_dir,
            state_dir=state_dir,
            seed=seed,
            device=device,
            num_clients=num_clients,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            num_parallels=num_parallels,
        )
    pipeline = FedAvgPipeline(handler=handler, trainer=trainer)
    try:
        pipeline.main()
    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt")
    except Exception as e:
        logging.error(e)


if __name__ == "__main__":
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="cnn")
    parser.add_argument("--num_clients", type=int, default=100)
    parser.add_argument("--global_round", type=int, default=5)
    parser.add_argument("--sample_ratio", type=float, default=1.0)
    parser.add_argument("--partition", type=str, default="shards")
    parser.add_argument("--num_shards", type=int, default=200)
    parser.add_argument("--dir_alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--num_parallels", type=int, default=10)
    parser.add_argument(
        "--dataset_root_dir", type=str, default="/tmp/quickstart-fedavg/dataset"
    )
    parser.add_argument(
        "--dataset_split_dir", type=str, default="/tmp/quickstart-fedavg/split"
    )
    parser.add_argument("--share_dir", type=str, default="/tmp/quickstart-fedavg/share")
    parser.add_argument("--state_dir", type=str, default="/tmp/quickstart-fedavg/state")
    parser.add_argument("--serial", action="store_true")

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    dataset_root_dir = Path(args.dataset_root_dir)
    dataset_split_dir = dataset_root_dir.joinpath(timestamp)
    share_dir = Path(args.share_dir).joinpath(timestamp)
    state_dir = Path(args.state_dir).joinpath(timestamp)
    state_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    main(
        model_name=args.model_name,
        num_clients=args.num_clients,
        global_round=args.global_round,
        sample_ratio=args.sample_ratio,
        partition=args.partition,
        num_shards=args.num_shards,
        seed=args.seed,
        dir_alpha=args.dir_alpha,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        num_parallels=args.num_parallels,
        dataset_root_dir=dataset_root_dir,
        dataset_split_dir=dataset_split_dir,
        share_dir=share_dir,
        state_dir=state_dir,
        device=device,
        serial=args.serial,
    )
