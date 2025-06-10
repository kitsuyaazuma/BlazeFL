from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Protocol, TypeVar

from tqdm import tqdm

UplinkPackage = TypeVar("UplinkPackage")
DownlinkPackage = TypeVar("DownlinkPackage", contravariant=True)


class MultiThreadClientTrainer(Protocol[UplinkPackage, DownlinkPackage]):
    num_parallels: int
    device: str
    device_count: int
    cache: list[UplinkPackage]

    def process_client(
        self,
        cid: int,
        device: str,
        payload: DownlinkPackage,
    ) -> UplinkPackage: ...

    def get_client_device(self, cid: int) -> str:
        if self.device == "cuda":
            return f"cuda:{cid % self.device_count}"
        return self.device

    def local_process(self, payload: DownlinkPackage, cid_list: list[int]) -> None:
        with ThreadPoolExecutor(max_workers=self.num_parallels) as executor:
            futures = []
            for cid in cid_list:
                device = self.get_client_device(cid)
                future = executor.submit(
                    self.process_client,
                    cid,
                    device,
                    payload,
                )
                futures.append(future)

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Client", leave=False
            ):
                result = future.result()
                self.cache.append(result)
