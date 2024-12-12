import multiprocessing as mp


def pytest_sessionstart() -> None:
    mp.set_start_method("spawn", force=True)
