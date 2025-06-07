from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generic, TypeVar

import torch
from blazefl.core import SerialClientTrainer
from tqdm import tqdm

UplinkPackage = TypeVar("UplinkPackage")
DownlinkPackage = TypeVar("DownlinkPackage")
ClientConfig = TypeVar("ClientConfig")


class MultithreadClientTrainer(
    SerialClientTrainer, Generic[UplinkPackage, DownlinkPackage, ClientConfig]
):
    def __init__(self, num_parallels: int, device: str) -> None:
        self.num_parallels = num_parallels
        self.device = device
        if self.device == "cuda":
            self.device_count = torch.cuda.device_count()
        self.cache: list[UplinkPackage] = []

    @abstractmethod
    def process_client(
        self, cid: int, device: str, payload: DownlinkPackage, config: ClientConfig
    ) -> UplinkPackage:
        pass

    @abstractmethod
    def get_client_config(self, cid: int) -> ClientConfig:
        pass

    def get_client_device(self, cid: int) -> str:
        if self.device == "cuda":
            return f"cuda:{cid % self.device_count}"
        return self.device

    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None:
        with ThreadPoolExecutor(max_workers=self.num_parallels) as executor:
            futures = []
            for cid in cid_list:
                config = self.get_client_config(cid)
                device = self.get_client_device(cid)
                future = executor.submit(
                    self.process_client, cid, device, payload, config
                )
                futures.append(future)

            for future in tqdm(as_completed(futures), desc="Client", leave=False):
                result = future.result()
                self.cache.append(result)
