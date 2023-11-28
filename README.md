# Online Supplement of "Dynamic Scheduling of a Multiclass Queue in the Halfin-Whitt Regime: A Computational Approach for High-Dimensional Problems"

This repository contains the codes for the computational method (Section 5), data, test problems and benchmark policies (Section 6), and numerical results (Section 7) of the paper "Dynamic Scheduling of a Multiclass Queue in the Halfin-Whitt Regime: A Computational Approach for High-Dimensional Problems."

## Folders

### `data`:
  * `MayJulyWeekdays`: Data used to estimate system parameters. This data comes from the publicly available [US Bank Call Center dataset](https://see-center.iem.technion.ac.il/databases/USBank/) provided by the Service Enterprise Engineering Lab at Technion. 
  * `csv_export.py: Code for exporting database tables to CSV.
  * `Test_Problems_Data_Analysis.ipynb`: Jupyter Notebook for estimating system parameters for each customer class.
 
### `bsde_solver`:
   * The deep neural network code for solving the HJB equation in our stochastic control problem.

### `secondary_analysis`: 
   * Codes for generating benchmark policies and simulating the neural network policy.
   * `simulation_codes`:
      * `nn_simulation`: contains the C++ code used to simulate the policy proposed by the neural network. 
      * `ctmc_simulation`: contains the C++ code used to simulate the optimal policy given by the CTMC solution.
      * `benchmark_simulation`: contains the C++ code used to simulate the static priority rules used as the benchmark policies for the main and high dimensional test problems. 
   * `mdp_solution`: contains the code used to solve the associated CTMC in low dimensional test problems to generate the benchmark policy.
 
## Running Experiments

### To run experiments related to neural network policy in the paper

* Generate the system parameters of the queueing system by running `Test_Problems_Data_Analysis.ipynb`
* Run `nn_main.jl` code to solve the HJB equation of the associated control problem and save neural network weights. 
* Simulate the neural network policy using the `nn_sim.cpp` code in `secondary_analysis/simulation_codes/nn_simulation/` folder. 

## To run experiments related to the benchmark policy generation for low dimensional problems in the paper

* Generate the system parameters of the queueing system by running `Test_Problems_Data_Analysis.ipynb`
* Run `main.m` code to solve the associated CTMC and save the optimal policies.
* Simulate the optimal policy using `ctmc_sim.cpp` in `secondary_analysis/simulation_codes/ctmc_simulation/`.

## To run experiments related to benchmark policy generation for high dimensional problems in the paper

* Generate the system parameters of the queueing system by running `Test_Problems_Data_Analysis.ipynb`
* Simulate the heuristic policies using `benchmark_sim.cpp` in `secondary_analysis/simulation_codes/benchmark_simulation/`.



