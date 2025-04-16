Benchmarks
==========

Benchmarks were conducted using Google Cloud’s Compute Engine with the following specifications:

Machine Configuration
----------------------

- **Machine Type**: `a2-highgpu-1g <https://cloud.google.com/compute/docs/gpus#a2-standard>`_ (vCPU count: 12, VM memory: 85 GB)
- **CPU Platform**: Intel Cascade Lake
- **GPU**: 1 x NVIDIA A100 40GB
- **Boot Disk**: 250 GB SSD

Benchmark Setup
---------------

- **Algorithm**: `FedAvg <https://proceedings.mlr.press/v54/mcmahan17a>`_
- **Dataset**: `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_
- **Number of Clients**: 100
- **Communication Rounds**: 5
- **Local Training**: 5 epochs, Learning Rate: 0.1, Batch Size: 50
- **Role**:
  - Server: Aggregation
  - Clients: Training and Evaluation (80% training, 20% evaluation)
- **Models**:
  - `CNN <https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html>`_ (size: 0.24 MB)
  - `ResNet18 <https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html>`_ (size: 44.59 MB)

For benchmarking purposes, we utilized Flower’s `Quickstart Example <https://github.com/adap/flower/tree/main/examples/quickstart-pytorch>`_ as a baseline to evaluate BlazeFL’s performance and efficiency.

.. raw:: html

   <div style="display: flex; justify-content: center; gap: 10px; align-items: center;">
       <img src="https://raw.githubusercontent.com/kitsuyaazuma/BlazeFL/refs/heads/main/docs/imgs/benchmark_cnn.png"
            alt="CNN Benchmark" style="width: 45%;">
       <img src="https://raw.githubusercontent.com/kitsuyaazuma/BlazeFL/refs/heads/main/docs/imgs/benchmark_resnet18.png"
            alt="ResNet18 Benchmark" style="width: 45%;">
   </div>
