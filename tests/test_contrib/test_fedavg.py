import os
import signal
import time
from contextlib import suppress
from multiprocessing import Process

import psutil
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.blazefl.contrib.fedavg import (
    FedAvgParallelClientTrainer,
    FedAvgSerialClientTrainer,
    FedAvgServerHandler,
)
from src.blazefl.core import ModelSelector, PartitionedDataset


class DummyModelSelector(ModelSelector):
    def select_model(self, model_name: str) -> torch.nn.Module:
        _ = model_name
        return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))


class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.data = torch.randn(size, 2, 2)
        self.targets = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class DummyPartitionedDataset(PartitionedDataset):
    def __init__(self, num_clients: int, size_per_client: int):
        self.num_clients = num_clients
        self.datasets = []
        for cid in range(num_clients):
            g = torch.Generator().manual_seed(cid)
            data = torch.randn((size_per_client, 2, 2), generator=g)
            targets = torch.randint(0, 2, (size_per_client,), generator=g)
            dataset = torch.utils.data.TensorDataset(data, targets)
            self.datasets.append(dataset)

    def get_dataset(self, type_: str, cid: int | None):
        _ = type_
        if cid is None:
            cid = 0
        return self.datasets[cid]

    def get_dataloader(self, type_: str, cid: int | None, batch_size: int | None):
        dataset = self.get_dataset(type_, cid)
        if batch_size is None:
            batch_size = len(dataset)
        return DataLoader(dataset, batch_size=batch_size)


@pytest.fixture
def model_selector():
    return DummyModelSelector()


@pytest.fixture
def partitioned_dataset():
    return DummyPartitionedDataset(num_clients=3, size_per_client=10)


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def tmp_share_dir(tmp_path):
    share_dir = tmp_path / "share"
    return share_dir


@pytest.fixture
def tmp_state_dir(tmp_path):
    state_dir = tmp_path / "state"
    return state_dir


def test_server_and_serial_integration(model_selector, partitioned_dataset, device):
    model_name = "dummy"
    global_round = 1
    num_clients = 3
    sample_ratio = 1.0
    epochs = 1
    batch_size = 2
    lr = 0.01

    server = FedAvgServerHandler(
        model_selector=model_selector,
        model_name=model_name,
        dataset=partitioned_dataset,
        global_round=global_round,
        num_clients=num_clients,
        sample_ratio=sample_ratio,
        device=device,
    )

    trainer = FedAvgSerialClientTrainer(
        model_selector=model_selector,
        model_name=model_name,
        dataset=partitioned_dataset,
        device=device,
        num_clients=num_clients,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )

    cids = server.sample_clients()
    assert len(cids) == num_clients
    downlink = server.downlink_package()
    trainer.local_process(downlink, cids)
    uplinks = trainer.uplink_package()
    assert len(uplinks) == num_clients

    done = False
    for pkg in uplinks:
        done = server.load(pkg)
    assert done is True
    assert server.round == 1

    assert server.if_stop() is True


def test_server_and_parallel_integration(
    model_selector, partitioned_dataset, device, tmp_share_dir, tmp_state_dir
):
    model_name = "dummy"
    global_round = 2
    num_clients = 3
    sample_ratio = 1.0
    epochs = 1
    batch_size = 2
    lr = 0.01
    seed = 42
    num_parallels = 2

    server = FedAvgServerHandler(
        model_selector=model_selector,
        model_name=model_name,
        dataset=partitioned_dataset,
        global_round=global_round,
        num_clients=num_clients,
        sample_ratio=sample_ratio,
        device=device,
    )

    trainer = FedAvgParallelClientTrainer(
        model_selector=model_selector,
        model_name=model_name,
        share_dir=tmp_share_dir,
        state_dir=tmp_state_dir,
        dataset=partitioned_dataset,
        device=device,
        num_clients=num_clients,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        num_parallels=num_parallels,
    )

    for round_ in range(1, global_round + 1):
        cids = server.sample_clients()
        downlink = server.downlink_package()
        trainer.local_process(downlink, cids)
        uplinks = trainer.uplink_package()
        assert len(uplinks) == num_clients

        done = False
        for pkg in uplinks:
            done = server.load(pkg)
        assert done is True
        assert server.round == round_

    assert server.if_stop() is True


def run_local_process(trainer, downlink, cids):
    with suppress(KeyboardInterrupt):
        trainer.local_process(downlink, cids)


def test_server_and_parallel_integration_keyboard_interrupt(
    model_selector, partitioned_dataset, device, tmp_share_dir, tmp_state_dir
):
    model_name = "dummy"
    global_round = 1
    num_clients = 10
    sample_ratio = 1.0
    epochs = 10**5
    batch_size = 2
    lr = 0.01
    seed = 42
    num_parallels = 10

    server = FedAvgServerHandler(
        model_selector=model_selector,
        model_name=model_name,
        dataset=partitioned_dataset,
        global_round=global_round,
        num_clients=num_clients,
        sample_ratio=sample_ratio,
        device=device,
    )

    trainer = FedAvgParallelClientTrainer(
        model_selector=model_selector,
        model_name=model_name,
        share_dir=tmp_share_dir,
        state_dir=tmp_state_dir,
        dataset=partitioned_dataset,
        device=device,
        num_clients=num_clients,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        num_parallels=num_parallels,
    )

    cids = server.sample_clients()
    downlink = server.downlink_package()

    proc = Process(target=run_local_process, args=(trainer, downlink, cids))
    proc.start()
    assert proc.pid is not None

    spawned_pids = []
    timeout = 5
    start_time = time.time()
    while proc.is_alive():
        spawned_pids = []
        for child in psutil.Process(proc.pid).children(recursive=True):
            spawned_pids.append(child.pid)
        if len(spawned_pids) == num_parallels:
            break
        if time.time() - start_time > timeout:
            raise AssertionError("Timeout reached while waiting for spawned processes.")
    assert proc.is_alive()

    os.kill(proc.pid, signal.SIGINT)

    proc.join(timeout=5)
    assert not proc.is_alive()

    orphan_pids = []
    for pid in spawned_pids:
        if psutil.pid_exists(pid):
            orphan_pids.append(pid)
    assert len(orphan_pids) == 0
