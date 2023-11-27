# CTMC-Based Discrete Event Simulation

## Overview

This repository contains the source code for 2-dimensional and 3-dimensional continuous-time Markov chain (CTMC) simulations. It is implemented in C++ and utilizes OpenMP for parallel computation. The simulation is designed to model a call center (with the preemptive resume scheduling rule) where the solution of the associated CTMC determines optimal priority policy.  

## Key Components
- 'Simulation' Class: This class is responsible for setting up and running the simulation. It reads configuration settings from a JSON file.
- 'Execute' Class: This class manages the core logic of the CTMC simulation, including handling different events and updating the system state based on the optimal priority rule determined by the CTMC solution.

## Initialization
- The 'Simulation' constructor initializes the simulation using a JSON configuration file. It contains key simulation parameters like num_interval, num_iterations, and file paths for various input data.

- Data Loading: Various parameters and data sets are loaded from CSV files. These include arrival rates (lambda), service rates (mu_hourly), and other operational parameters.

## Running the Simulation
- 'main' Function: Creates a 'Simulation' object with the specified configuration file and runs the simulation.
- Output: The 'save' method of the 'Simulation' class outputs the results to a CSV file, as specified by 'record_file'.

## Event Handling
The 'Execute' class includes methods to handle different events in the simulation: 
- Arrival Events: 'handle_arrival_event' updates the system upon new arrivals.
- Departure Events: 'handle_depart_event' manages departures from the system.
- Abandonment Events: 'handle_abandon_event' handles situations where a customer leaves the queue before being served.

## Optimal Policy Calculation
The simulation finds the optimal policies based on the current system state and time interval from an optimal policy matrix (which comes from the MATLAB code provided in the dp_low_dimensional folder). This is managed by 'optimal_policy_calculation'. We use the preemptive resume scheduling rule. Please refer to the dp_low_dimensional folder for how we determine the optimal policy given state and time interval.

## Simulation Flow
The 'run' method in the 'Execute' class determines the simulation flow, advancing time, and processing events accordingly. 

## Output
The simulation calculates and outputs the costs incurred by customers waiting in the queue during a 17-hour daily operation at a call center. The num_iterations parameter determines the number of days simulated. We save the results to 'record_file'. 

## Technical Details

### Prerequisites
- C++11 compiler (e.g., GCC, Clang)
- CMake (version 3.20 or higher)
- OpenMP

### Structure
- ctmc_sim.cpp: The main simulation code.
- ctmc_sim.h: Header file for the simulation.
- CMakeLists.txt: CMake configuration file for building the project.

### Compilation and Running

#### Clone the Repository
To clone the repository, use the following command:
```bash
git clone [repository URL]
cd [repository directory]
```

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

#### Configuration 
The simulation is configured using a JSON file. Please edit the 'config.json' file to adjust parameters such as 'num_interval', 'num_iterations', 'decision_freq', etc. Also, please edit the 'record_file' to save the output.





