# Low-dimensional CTMC Solution

## Overview

This MATLAB code solves the 2D and 3D continuous-time Markov chain (CTMC) models for our call center. It computes the optimal policy for a given system state and time.

## Key Features
- Calculation of the value function V(X(t),t) and optimal policy for 2D and 3D CTMCs.
- Use of JSON for configuration settings and data preprocessing steps.
- Outputs optimal policies and value functions at the intervals specified by the user for application in the simulation of the call center operations (please see the [ctmc_simulation](https://github.com/ekasikaralar/NN_based_dynamic_scheduling/tree/main/secondary_analysis/simulation_codes/ctmc_simulation) folder for the code used to simulate the optimal policy).

### Function Descriptions
'main' (main_2dim or main_3dim):
- Initializes file paths for recording policies and value functions.
- Reads configuration data from JSON and preprocesses the input data.
- Calls 'runCTMC' to start the CTMC solution.

'defineConfig':
- Defines configuration settings and parameters

'readAndPreprocessData':
- Reads input data like arrival, service, and cost rates from CSV files.
- Scales data according to the time discretization.

'runCTMC'
- Solves the CTMC problem and returns the value function V(X,t=0).
- Iteratively computes the Hamilton-Jacobi-Bellman (HJB) equation.
- Records the policy matrix and value function at specified intervals.

## How to Run
Please run the following command to start the MATLAB code:

```bash
      export OMP_THREAD_LIMIT=128 ## for paralleling the code
      matlab -nodisplay -nosplash -nodesktop -r "main_2dim; exit;" #(for 2D problem)
      #matlab -nodisplay -nosplash -nodesktop -r "main_3dim; exit;" #(for 3D problem)
```


## Configuration and Customization
Please run main_2dim.m for 2D problems and main_3dim.m for 3D problems. Additionally, please
- Modify the 'config.json' file
- Update the file paths in the 'main' function for different input data or output locations.
