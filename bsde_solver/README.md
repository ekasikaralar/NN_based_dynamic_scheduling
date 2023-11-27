# Neural Network (BSDE Solver)

## Overview

The code is intended for the implementation and training of the deep neural network to determine the optimal policy by solving the HJB equation in our stochastic control problem.
It is structured to be used with a configuration file, 'config.json', which includes parameters for the neural network and the system describing the call center we are interested in.

## Prerequisites

To run this code, you need Julia (1.9) installed with the following packages:

- CUDA
- cuDNN
- Random
- Flux
- DataFrames
- StatsBase
- JLD2
- JSON
- CSV
- TickTock
- DelimitedFiles
- NPZ
- Distributions


## Configuration

The 'config.json' file should contain two primary sections:

1. 'neural_network_parameters': Specifies parameters like the learning rate, number of neurons, batch sizes, etc.
2. 'system_parameters': Includes parameters related to the system that the neural network will interact with, such as service rates, cost rates, and other operational parameters.


## Key Components of the Code

1. CUDA Configurations: Sets up CUDA for GPU-based computations.
2. Parameter Initialization: Reads parameters from the configuration file.
3. System Parameters: Reads system-related parameters and adjusts them according to the precision required.
4. Neural Network Construction: Provides functions to create deep neural network chains with specific configurations.
5. Model Definition: Defines 'NonsharedMode' struct, encapsulating the neural networks for estimating the value function and its gradient.
6. Sampling Function: Functions for generating training and validation samples.

## Usage

1.  Set up Configuration File: Before running the code, please make sure that 'config.json' is correctly set up with all necessary parameters.
2.  Set up the RUN_NAME: The main training code (nn_main.jl) has a RUN_NAME variable used to specify the name of the folder, where we save neural network weights at the end of the training. Please change the RUN_NAME according to your setting.
4.  Run the Script: Execute the script in a Julia environment. Specifically, run the command: 

```bash
julia nn_main.jl
```

## Important Notes

- This code is structured to use GPU acceleration for faster computation. Please ensure that your system supports CUDA and has a compatible GPU.
- Modify the 'config.json' file and the RUN_NAME (OLD_NAME) as per your specific requirements before running the script.
  
- Additionally, we provide **nn_main_cont.jl**. This continuation code is designed for resuming or extending the training of a neural network from a previously saved state. This is crucial in scenarios where:
  - Long Training Durations: Initial training might have been interrupted or stopped due to time constraints or system limitations.
  - Refinement: After evaluating the performance of the initially trained model, you might identify the need for further training to refine the model's accuracy.
  
  ## How the Continuation Code Works
  - Loading Previous States: It starts by loading the neural network weights and optimizer states saved from a previous training session.
  - Seed Adjustment: The random seed is set to a new value ('Random.seed!(74)'), ensuring that the new training phase introduces some variation in the training data.
  - Adjustment of a_lowbound: We assume a_lowbound converges to 0.0 after an initial training.

  ## Usage

  1. Set up the OLD_NAME and RUN_NAME: OLD_NAME should match the name of the folder that contains the neural network weights and optimizer saved after the initial training. RUN_NAME should be the name of the folder for the continuation training.
  2. Run the Script: Execute the script in Julia as follows:
      
     ```bash
      julia nn_main_cont.jl
     ```
     
This code is adapted from https://github.com/frankhan91/DeepBSDE
