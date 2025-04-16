Overview
================

Why Choose BlazeFL?
-------------------

- 🚀 **High Performance**: Optimized for single-node simulations, BlazeFL allows you to adjust the degree of parallelism. For example, if you want to simulate 100 clients on a single node but lack the resources to run them all concurrently, you can configure 10 parallel processes to manage the simulation efficiently.

- 🔧 **Extensibility**: BlazeFL provides interfaces solely for communication and parallelization, avoiding excessive abstraction. This design ensures that the framework remains flexible and adaptable to various use cases.

- 📦 **Minimal Dependencies**: The core components of BlazeFL rely only on `PyTorch <https://github.com/pytorch/pytorch>`_, ensuring a lightweight and straightforward setup.

- 🔄 **Robust Reproducibility**: Even in multi-process environments, BlazeFL offers utilities to save and restore seed states, ensuring consistent and reproducible results across simulations.

- 🏷️ **Type Hint Support**: The framework fully supports type hints, enhancing code readability and maintainability.

- 🔗 **Loose Compatibility with FedLab**: Inspired by `FedLab <https://github.com/SMILELab-FL/FedLab>`_, BlazeFL maintains a degree of compatibility, facilitating an easy transition to production-level implementations when necessary.

How BlazeFL Works
-------------------

BlazeFL enhances performance by storing shared parameters on disk instead of shared memory, enabling efficient parameter sharing across processes, simplifying memory management, and reducing overhead.

.. image:: https://raw.githubusercontent.com/kitsuyaazuma/BlazeFL/refs/heads/main/docs/imgs/architecture.png
   :align: center


