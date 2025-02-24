# Discrete Event Simulation for Appendix B.2.3 (Additional static priority policies for the main test problem and its two variants)

## Overview

This repository contains the source code for the discrete event simulation used to generate the additional static priority policies in Appendix B.2.3. The simulation models a queueing system where different customer classes arrive, receive service, and potentially abandon the system. It is implemented in **C++** and leverages **OpenMP** for parallel computation.

This simulation evaluates **125 different priority policies**. 

## Key Components

### `Simulation` Class
- Reads simulation configuration from a JSON file.
- Loads system parameters such as arrival rates, service rates, and cost structures from CSV files.
- Generates 125 different policy combinations for evaluation.
- Runs the simulation multiple times as defined by `num_iterations`.

### `Execute` Class
- Implements the core logic of the discrete event simulation.
- Handles **arrival**, **service completion**, and **abandonment** events.
- Applies preemptive scheduling rules to manage queue priorities dynamically.
- Computes **waiting costs**, **holding costs**, and **total system cost**.

## Configuration & Initialization

The simulation requires a JSON configuration file that defines:
- **num_interval**: The number of time intervals per day.
- **num_iterations**: The number of simulation replications (days).
- **File paths** for input data, such as arrival rates, service rates, and cost parameters.

The following input data is loaded from CSV files:
- **lambda.csv** → Arrival rates per 5-minute interval.
- **agents.csv** → Number of servers over time.
- **mu_hourly.csv** → Hourly service rates.
- **theta_hourly.csv** → Hourly abandonment rates.
- **arr_cdf.csv** → Arrival cumulative distribution function (CDF).
- **holding_cost_rate.csv** → Hourly holding costs.
- **abandonment_cost_rate.csv** → Hourly abandonment costs.
- **cost_rate.csv** → Hourly system costs.
- **initialization.csv** → Initial system state.

## Static Priority Policy Evaluation

The simulation evaluates **125 different static priority policies**, generated from the following base policies:
1. **c_mu**
2. **cost**
3. **c_mu_theta**
4. **c_mu_theta_diff**
5. **mu_theta_diff**

Each policy is evaluated based on three priority tiers, forming all possible combinations of the five base policies.

## Running the Simulation

### Compilation
Ensure you have a **C++11-compatible compiler**, OpenMP, and CMake installed.

```bash
mkdir build
cd build
cmake ..
make
```

### Running the Simulation
```bash
export OMP_THREAD_LIMIT=800
./simulation
```

## Technical Details

### Parallelization
- **OpenMP** is used for parallel execution to process 10,000 iterations efficiently.
- The **policy evaluations** are executed in parallel to reduce runtime.

### Key Functions
- `readMatrixFromCSV()`, `readVectorFromCSV()`: Read input data.
- `generate_interarrival()`, `generate_abandon()`, `generate_service()`: Generate random events.
- `handle_arrival_event()`, `handle_depart_event()`, `handle_abandon_event()`: Process events.
- `queueing_discipline()`: Determines priority-based scheduling decisions.

## Prerequisites
- **C++ compiler** (GCC, Clang)
- **CMake 3.20+**
- **OpenMP** (for multi-threading)
- **JSON for Modern C++** (`nlohmann/json.hpp`)
