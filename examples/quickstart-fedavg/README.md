# Quickstart: Federated Averaging (FedAvg)

Welcome to the quickstart guide for the Federated Averaging (FedAvg [^1]) with BlazeFL!

## Setup

### 1. Clone the repository

First, clone the BlazeFL repository and navigating to the `quickstart-fedavg` directory:

```bash
git clone https://github.com/kitsuyaazuma/blazefl.git
cd blazefl/examples/quickstart-fedavg
```

### 2. Install dependencies

Install the necessary dependencies using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

## Running the example

To run the FedAvg example, use the following command:

```bash
uv run python main.py num_parallels=5
```

Adjust the `num_parallels` parameter to suit your system’s specifications for optimal performance.

For more options and configurations, see the `config.yaml` file. All hyperparameters are managed with Hydra and located under the `config/` directory. You can override any setting directly from the command line:

```bash
uv run python main.py partition=client_inner_dirichlet dir_alpha=0.5
```


[^1]: B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y. Arcas, "Communication-Efficient Learning of Deep Networks from Decentralized Data," in Proc. 20th Int. Conf. Artif. Intell. Stat., ser. Proc. Mach. Learn. Res., A. Singh and J. Zhu, Eds., vol. 54. PMLR, 2017, pp. 1273–1282.
