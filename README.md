# Online Supplement of "Dynamic Scheduling of a Multiclass Queue in the Halfin-Whitt Regime: A Computational Approach for High-Dimensional Problems"

This repository contains the codes for the computational method (Section 5), data, test problems and benchmark policies (Section 6), and numerical results (Section 7) of the paper "Dynamic Scheduling of a Multiclass Queue in the Halfin-Whitt Regime: A Computational Approach for High-Dimensional Problems."

## Folders

### `data`:
  * `MayJulyWeekdays`: Data used to estimate system parameters. This data comes from the publicly available [US Bank Call Center dataset](https://see-center.iem.technion.ac.il/databases/USBank/) provided by the Service Enterprise Engineering Lab at Technion. 
  * `csv_export.py`: Code for exporting database tables to CSV.
  * `Test_Problems_Data_Analysis.ipynb`: Jupyter Notebook for estimating system parameters for each customer class.
 
### `bsde_solver`:
   * The deep neural network code for solving the HJB equation in our stochastic control problem.

### `secondary_analysis`: 
   * Codes for generating benchmark policies and simulating the neural network policy.
   `simulation_codes`:
      * `nn_simulation`: C++ code for simulating neural network policy. 
      * `ctmc_simulation`: C++ code for simulating the optimal CTMC solution policy.
      * `benchmark_simulation`: C++ code for simulating static priority rule benchmarks. 
      * `mdp_solution`: Code for solving the associated CTMC in low dimensional test problems.
 
## Running Experiments

### To run experiments related to neural network policy in the paper

* Run `Test_Problems_Data_Analysis.ipynb` to generate system parameters.
* Execute `nn_main.jl` to solve the HJB equation and save neural network weights.
* Simulate policy using `nn_sim.cpp` in `/secondary_analysis/simulation_codes/nn_simulation/`.

### To run experiments related to the benchmark policy generation for low dimensional problems in the paper

* Run `Test_Problems_Data_Analysis.ipynb` to generate system parameters.
* Run `main.m` code to solve the associated CTMC and save the optimal policies.
* Simulate the optimal policy using `ctmc_sim.cpp` in `/secondary_analysis/simulation_codes/ctmc_simulation/`.

### To run experiments related to benchmark policy generation for high dimensional problems in the paper

* Run `Test_Problems_Data_Analysis.ipynb` to generate system parameters.
* Simulate the heuristic policies using `benchmark_sim.cpp` in `/secondary_analysis/simulation_codes/benchmark_simulation/`.



