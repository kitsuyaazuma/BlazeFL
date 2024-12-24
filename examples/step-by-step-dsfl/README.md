# Step-by-Step Tutorial: DS-FL

Welcome to this step-by-step tutorial on implementing [DS-FL](https://doi.ieeecomputersociety.org/10.1109/TMC.2021.3070013) using BlazeFL. DS-FL is a Federated Learning method that performs knowledge distillation by exchanging model outputs on an open dataset. 

Thanks to its high extensibility, the BlazeFL framework allows for the simple implementation of not only typical parameter exchange methods but also distillation-based methods and other innovative approaches, much like assembling puzzle pieces. 

This tutorial will surely assist you in implementing your unique and unconventional FL methods, helping you break free from traditional constraints.

## Setup a Project

First, open your terminal and create a new directory for your project.

```bash
mkdir step-by-step-dsfl
cd step-by-step-dsfl
```

Next, Initialize the projcet with [uv](https://github.com/astral-sh/uv) or your favorite package manager.

```bash
uv init
```

Then, create a virtual environment and install BlazeFL. 

```bash
uv venv --python 3.12
source .venv/bin/activate
uv add blazefl
```

## Implement the Algorithm

Now, let's implement the DS-FL algorithm. 

### Define the Partitioned Dataset

```bash
uv add torchvision fedlab scikit-learn
```




## Run the Simulation
