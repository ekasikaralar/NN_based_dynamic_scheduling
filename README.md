# Online Supplement of "Dynamic Scheduling of a Multiclass Queue in the Halfin-Whitt Regime: A Computational Approach for High-Dimensional Problems"

This repository contains the codes for the computational method (Section 5), data, test problems and benchmark policies (Section 6), and numerical results (Section 7) of the paper "Dynamic Scheduling of a Multiclass Queue in the Halfin-Whitt Regime: A Computational Approach for High-Dimensional Problems."

## Proposed Procedures

Folders and their contents

* `data`:
* * `MayJulyWeekdays` is the data we use to estimate the system parameters. This data comes from the publicly available [US Bank Call Center dataset](https://see-center.iem.technion.ac.il/databases/USBank/) provided by the Service Enterprise Engineering Lab at Technion. 
* * `csv_export.py` is the code to export specific tables from database files (MDB format) to CSV files.
* * `Test_Problems_Data_Analysis.ipynb` is the Juypter Notebook to estimate the system parameters for each customer class in our queueing system.
  * 
