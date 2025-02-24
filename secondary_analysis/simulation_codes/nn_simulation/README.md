# Neural Network-Based Discrete Event Simulation

## Overview

This repository contains the source code for neural network-based discrete event simulations. It is implemented in C++ and utilizes OpenMP for parallel computation. The simulation is designed to model a call center (with the preemptive resume scheduling rule) where neural networks determine the priority policy (please see [deep_splitting](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/deep_splitting) code for the associated neural network code).

## Key Components
- `Simulation` Class: This class is responsible for setting up and running the simulation. It reads configuration settings from a JSON file.
- `Execute` Class: This class manages the core logic of the simulation, including handling different events and updating the system state based on the optimal priority rule determined by the neural networks.
- `MyNetwork` Class: This class represents the neural network model to generate the gradient approximations. It loads the neural network weights (saved using the Julia code in the [deep_splitting](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/deep_splitting) folder) saved under 'neural_network_folder_name' and includes methods for forward propagation, batch normalization, and performing matrix operations.  

## Initialization
- The `Simulation` constructor initializes the simulation using a JSON configuration file. It contains key simulation parameters like num_interval, num_iterations, and file paths for various input data.

- Data Loading: Various parameters and data sets are loaded from CSV files. These include arrival rates (lambda), service rates (mu_hourly), and other operational parameters.

## Running the Simulation
- `main` Function: Creates a `Simulation` object with the specified configuration file and runs the simulation.
- Output: The `save` method of the `Simulation` class outputs the results to a CSV file, as specified by `record_file`.

## Neural Network Integration
- The simulation integrates a neural network model to determine the priority policy. This is a key aspect of the simulation, enabling dynamic decision-making based on the current state of the system and a given time interval. 
- The `MyNetwork` class handles the loading of the neural network parameters, execution of forward passes, and application of matrix operations for optimal policy determination.

## Policy Calculation
- The `Execute` class is critical in this process, utilizing the neural network's output to determine the proposed priority rule.
- Specifically, given the current state `num_in_system` and the time interval (determined by `decision_freq`), we scale the state `scaled_x` and use it as the input tensor to approximate the gradient of the value function.
- This approximation is subsequently utilized to calculate the effective holding cost rate, which determines the  priority order. For detailed operations, please refer to the `queueing_discipline` function in the `Execute` class.

## Output
The simulation calculates and outputs the costs incurred by customers waiting in the queue during a 17-hour daily operation at a call center. The `num_iterations` parameter determines the number of days simulated. We save the results to `record_file`. 

## Technical Details

### Prerequisites
- C++11 compiler (e.g., GCC, Clang)
- CMake (version 3.20 or higher)
- OpenMP for parallel processing

### Structure
- `nn_sim.cpp`: The main simulation code.
- `nn_sim.h`: Header file for the simulation and neural network classes.
- `CMakeLists.txt`: CMake configuration file for building the project.

### Compilation and Running

#### Build the Project
Use CMake and Make to build the project:
```bash
mkdir build
cd build
cmake ..
make
```

#### Run the Simulation
After building, run the simulation with: 
```bash
export OMP_THREAD_LIMIT=128
./prdes
```

#### Configuration 
The simulation is configured using a JSON file. Please edit the 'config.json' file to adjust parameters such as 'num_interval', 'num_iterations', 'decision_freq', etc. Also, please edit the 'record_file' to save the output.

