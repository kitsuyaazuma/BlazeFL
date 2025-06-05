import numpy as np
import numpy.typing as npt


def balance_split(num_clients: int, num_samples: int) -> npt.NDArray[np.int_]:
    """
    Adapted from https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
    """
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
    verbose: bool = True,
) -> tuple[dict[int, npt.NDArray[np.int_]], npt.NDArray[np.float64]]:
    """
    Adapted from https://github.com/SMILELab-FL/FedLab/blob/master/fedlab/utils/dataset/functional.py
    """
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
                while True:
                    new_class = np.random.randint(num_classes)
                    if class_amount[new_class] > 0:
                        curr_class = new_class
                        break
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = idx_list[
                curr_class
            ][class_amount[curr_class]]

            break

    client_dict = {cid: client_indices[cid] for cid in range(num_clients)}
    return client_dict, class_priors
