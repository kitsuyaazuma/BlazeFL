<div align="center"><img src="https://raw.githubusercontent.com/kitsuyaazuma/BlazeFL/refs/heads/main/docs/imgs/logo.svg" width=600></div>
<div align="center">A blazing-fast and lightweight simulation framework for Federated Learning</div>
<br>
<div align="center">
  <a href="https://github.com/astral-sh/uv"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json" alt="uv"></a>
  <a href="https://pypi.python.org/pypi/blazefl"><img src="https://img.shields.io/pypi/v/blazefl" alt="PyPI Version"></a>
  <a href="https://pypi.python.org/pypi/blazefl"><img src="https://img.shields.io/pypi/l/blazefl" alt="License"></a>
  <a href="https://pypi.python.org/pypi/blazefl"><img src="https://img.shields.io/pypi/pyversions/blazefl" alt="Python Versions"></a>
</div>


## Why Choose BlazeFL?

- 🚀 **High Performance**: Optimized for single-node simulations, BlazeFL allows you to adjust the degree of parallelism. For example, if you want to simulate 100 clients on a single node but lack the resources to run them all concurrently, you can configure 10 parallel processes to manage the simulation efficiently. 

- 🔧 **Extensibility**: BlazeFL provides interfaces solely for communication and parallelization, avoiding excessive abstraction. This design ensures that the framework remains flexible and adaptable to various use cases.

- 📦 **Minimal Dependencies**: Minimal Dependencies: The core components of BlazeFL rely only on [PyTorch](https://github.com/pytorch/pytorch), ensuring a lightweight and straightforward setup. 

- 🔄 **Robust Reproducibility**: Even in multi-process environments, BlazeFL offers utilities to save and restore seed states, ensuring consistent and reproducible results across simulations.

- 🏷️ **Type Hint Support**: The framework fully supports type hints, enhancing code readability and maintainability.

- 🔗 **Loose Compatibility with FedLab**: Inspired by [FedLab](https://github.com/SMILELab-FL/FedLab), BlazeFL maintains a degree of compatibility, facilitating an easy transition to production-level implementations when necessary.

## How BlazeFL Works

BlazeFL enhances performance by storing shared parameters on disk instead of shared memory, enabling efficient parameter sharing across processes, simplifying memory management, and reducing overhead.

<div align="center"><img src="https://raw.githubusercontent.com/kitsuyaazuma/BlazeFL/refs/heads/main/docs/imgs/architecture.png"></div>

## Quick Start

### Installation

BlazeFL is available on PyPI and can be installed using your preferred package manager.
 
For example:

```bash
uv add blazefl
# or
poetry add blazefl
# or
pip install blazefl
```

### Running Examples

Quick start code is in [examples/quickstart-fedavg](https://github.com/kitsuyaazuma/BlazeFL/tree/main/examples/quickstart-fedavg).

For a more detailed implementation guide, checkout the [examples/step-by-step-dsfl](https://github.com/kitsuyaazuma/BlazeFL/tree/main/examples/step-by-step-dsfl).


## FL Simulation Benchmarks

Benchmarks were conducted using Google Cloud’s Compute Engine with the following specifications:

<details>
<summary> Machine Configuration</summary>

- Machine Type: [a2-highgpu-1g](https://cloud.google.com/compute/docs/gpus#a2-standard) (vCPU count: 12, VM memory: 85 GB)
- CPU Platform: Intel Cascade Lake
- GPU: 1 x NVIDIA A100 40GB
- Boot Disk: 250 GB SSD
</details>


<details>
<summary> Benchmark Setup</summary>

- Algorithm: [FedAvg](https://proceedings.mlr.press/v54/mcmahan17a)
- Dataset: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- Number of Clients: 100
- Communication Rounds: 5
- Local Training: 5 epochs, Learning Rate: 0.1, Batch Size: 50
- Role
  -	Server: Aggregation
  -	Clients: Training and Evaluation (80% training, 20% evaluation)
- Models
  - [CNN](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) (size: 0.24 MB)
  - [ResNet18](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) (size: 44.59 MB)

</details>

For benchmarking purposes, we utilized Flower’s [Quickstart Example](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch) as a baseline to evaluate BlazeFL’s performance and efficiency.

<div style="display: flex; justify-content: center; align-items: center;">
  <img src="https://raw.githubusercontent.com/kitsuyaazuma/BlazeFL/refs/heads/main/docs/imgs/benchmark_cnn.png" alt="CNN" width="45%" />
  <img src="https://raw.githubusercontent.com/kitsuyaazuma/BlazeFL/refs/heads/main/docs/imgs/benchmark_resnet18.png" alt="ResNet18" width="45%" />
</div>


## Contributing

We welcome contributions from the community! If you'd like to contribute to this project, please follow these guidelines:

### Issues

If you encounter a bug, have a feature request, or would like to suggest an improvement, please open an issue on the GitHub repository. Make sure to provide detailed information about the problem or suggestion.

### Pull Requests

We gladly accept pull requests! Before submitting a pull request, please ensure the following:

1. Fork the repository and create your branch from main.
2. Ensure your code adheres to the project's coding standards.
3. Test your changes thoroughly.
4. Make sure your commits are descriptive and well-documented.
5. Update the README and any relevant documentation if necessary.

### Code of Conduct

Please note that this project is governed by our [Code of Conduct](https://github.com/kitsuyaazuma/BlazeFL/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report any unacceptable behavior.

Thank you for contributing to our project!
