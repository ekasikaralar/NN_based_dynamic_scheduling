# Neural Network (Deep Splitting Solver)

## Overview

This project involves the implementation of a deep splitting algorithm to solve high-dimensional differential equations using neural networks. It is structured for use with a configuration file `config.json`, which contains parameters for both the neural network and the system that describes the call center of interest.

## Configuration

The `config.json` file should contain two primary sections:

1. `neural_network_parameters`: Specifies parameters like the learning rate, number of neurons, batch sizes, etc.
2. `equation_parameters`: Includes parameters related to the system that the neural network will interact with, such as service rates, cost rates, and other operational parameters.


## Usage

1.  Set up Configuration File: Before running the code, please make sure that `config.json` is correctly set up with all necessary parameters.
2.  Run the Script: Execute the script in a `PyTorch` environment. Specifically, run the command: 

```bash
python3 main.py --config_path="config.json" --run_name="test_name"
```
