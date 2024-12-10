<p align="center"><img src="https://raw.githubusercontent.com/kitsuya0828/BlazeFL/refs/heads/main/docs/logo.svg" width=600></p>
<div align="center">
    <em>A blazing-fast and lightweight simulation framework for Federated Learning</em>
</div>

## Why Choose BlazeFL?

- 🚀 **High Performance**: Optimized for single-node simulations, BlazeFL allows you to adjust the degree of parallelism. For example, if you want to simulate 100 clients on a single node but lack the resources to run them all concurrently, you can configure 10 parallel processes to manage the simulation efficiently. Additionally, BlazeFL enhances performance by storing shared parameters on disk instead of using shared memory, simplifying memory management and reducing overhead. 

- 🔧 **Extensibility**: BlazeFL provides interfaces solely for communication and parallelization, avoiding excessive abstraction. This design ensures that the framework remains flexible and adaptable to various use cases.

- 📦 **Minimal Dependencies**: Minimal Dependencies: The core components of BlazeFL rely only on [PyTorch](https://github.com/pytorch/pytorch), ensuring a lightweight and straightforward setup. 

- 🔄 **Robust Reproducibility**: Even in multi-process environments, BlazeFL offers utilities to save and restore seed states, ensuring consistent and reproducible results across simulations.

- 🏷️ **Type Hint Support**: The framework fully supports type hints, enhancing code readability and maintainability.

- 🔗 **Loose Compatibility with FedLab**: Inspired by [FedLab](https://github.com/SMILELab-FL/FedLab), BlazeFL maintains a degree of compatibility, facilitating an easy transition to production-level implementations when necessary.

*Please note that BlazeFL is currently in beta.*

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

Example code is available in the [examples/quickstart-fedavg](https://github.com/kitsuya0828/BlazeFL/tree/main/examples/quickstart-fedavg) directory.


## FL Simulation Benchmarks

To be written.

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

Please note that this project is governed by our [Code of Conduct](https://github.com/kitsuya0828/BlazeFL/blob/main/CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report any unacceptable behavior.

Thank you for contributing to our project!
