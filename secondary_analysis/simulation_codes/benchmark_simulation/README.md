# Discrete Event Simulation for Static Priority Rules

## Overview

This repository contains the source code for the discrete event simulations of the main test problem. It is implemented in C++ and utilizes OpenMP for parallel computation. The simulation is designed to model a call center (with the preemptive resume scheduling rule) where a specified static priority rule determines the priority of the classes.  

We use this code to determine the static priority benchmarks for the main test problem and the pathwise priority benchmark for the high dimensional test problems.

## Key Components
- 'Simulation' Class: This class is responsible for setting up and running the simulation. It reads configuration settings from a JSON file.
- 'Execute' Class: This class manages the core logic of the discrete event simulation, including handling different events and updating the system state based on the specified static priority rule.

## Initialization
- The 'Simulation' constructor initializes the simulation using a JSON configuration file. It contains key simulation parameters like num_interval, num_iterations, and file paths for various input data.

- Data Loading: Various parameters and data sets are loaded from CSV files. These include arrival rates (lambda), service rates (mu_hourly), and other operational parameters.

## Running the Simulation
- 'main' Function: Creates a 'Simulation' object with the specified configuration file and runs the simulation.
- Output: The 'save' method of the 'Simulation' class outputs the results to a CSV file, as specified by 'record_file'.


## Priority Policy Calculation
The simulation includes several static priority rules 'c_mu_theta', 'c_mu_theta_diff', 'c_mu', 'cost' and 'mu_theta diff', which are used to determine the relative priority order of the classes when serving them in the system.


## Output
The simulation calculates and outputs the costs incurred by customers waiting in the queue during a 17-hour daily operation at a call center. The num_iterations parameter determines the number of days simulated. We save the results to 'record_file'. 

## Technical Details

### Prerequisites
- C++11 compiler (e.g., GCC, Clang)
- CMake (version 3.20 or higher)
- OpenMP for parallel processing

### Structure
- benchmark_sim.cpp: The main simulation code.
- benchmark_sim.h: Header file for the simulation.
- CMakeLists.txt: CMake configuration file for building the project.

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
./prdes
```
