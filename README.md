# Online Supplement of "Dynamic Scheduling of a Multiclass Queue in the Halfin-Whitt Regime: A Computational Approach for High-Dimensional Problems"

This repository contains the codes for the computational method (Section 5), data, test problems and benchmark policies (Section 6 and Appendix B.2), and numerical results (Section 7) of the paper "Dynamic Scheduling of a Multiclass Queue in the Halfin-Whitt Regime: A Computational Approach for High-Dimensional Problems."

## Folders

### `data`:
  * [MayJulyWeekdays](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/data/MayJulyWeekdays): Data used to estimate system parameters. This data comes from the publicly available [US Bank Call Center dataset](https://see-center.iem.technion.ac.il/databases/USBank/) provided by the Service Enterprise Engineering Lab at Technion. 
  * [Test_Problems_Data_Analysis.ipynb](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/blob/main/data/Test_Problems_Data_Analysis.ipynb): Jupyter Notebook for estimating system parameters for each customer class.
 
### `deep_splitting`:
   * The deep neural network code for solving the HJB equation in our stochastic control problem.

### `secondary_analysis`: 
   * Codes for generating benchmark policies and simulating the neural network policy.
   * [simulation_codes](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/secondary_analysis/simulation_codes):
      * [nn_simulation](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/secondary_analysis/simulation_codes/nn_simulation): C++ code for simulating neural network policy. 
      * [ctmc_simulation](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/secondary_analysis/simulation_codes/ctmc_simulation): C++ code for simulating the optimal CTMC solution policy and auxiliary CTMC policy described in Appendix B.2.4.
      * [benchmark_simulation](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/secondary_analysis/simulation_codes/benchmark_simulation): C++ code for simulating static priority rule benchmarks.
      * [extended_benchmark_simulations](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/secondary_analysis/simulation_codes/extended_benchmark_simulations): C++ code for simulating extended static priority rules described in Appendix B.2.3. 
   * [mdp_solution](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/secondary_analysis/mdp_solution): MATLAB Code for solving the associated CTMC in low dimensional test problems.
 
## Running Experiments

* Run [Test_Problems_Data_Analysis.ipynb](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/blob/main/data/Test_Problems_Data_Analysis.ipynb) to generate system parameters.

### To run experiments related to neural network policy in the paper

* Run `main.py` in [deep_splitting](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/deep_splitting) folder to solve the HJB equation and save neural network weights.
* Simulate policy using `nn_sim.cpp` in [nn_simulation](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/secondary_analysis/simulation_codes/nn_simulation) folder.

### To run experiments related to the benchmark policy for low dimensional problems in the paper

* Run `main.m` in [mdp_solution](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/secondary_analysis/mdp_solution) folder to solve the associated CTMC and save the optimal policies.
* Simulate the optimal policy using `ctmc_sim.cpp` in [ctmc_simulation](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/secondary_analysis/simulation_codes/ctmc_simulation) folder.

### To run experiments related to benchmark policies for the main test problem and high dimensional problems in the paper

* Simulate the heuristic policies using `benchmark_sim.cpp` in [benchmark_simulation](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/secondary_analysis/simulation_codes/benchmark_simulation) folder.

## Notes 
For instructions on running the codes, please refer to the README files in each respective folder.

