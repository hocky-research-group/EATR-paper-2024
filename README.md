# Rate Analysis Scripts

The Python code used to perform the rate analysis is located in `Software/rate_analysis`. The script is set up to run some selected analyses on the PLUMED colvar files from multiple sets of simulations to obtain rates from each set.

`Software/rate_analysis/params.json` is a JSON file containing some parameters for the analysis:

- "head\_dir": The path to the directory which contains all the data to be analyzed.
- "beta": The value of the thermodynamic inverse temperature $\beta=1/k_BT$.
- "columns": The column in the colvar files corresponding to:
  - "time": The simulation time
  - "bias": The instantaneous bias
  - "acc": The infrequent metadynamics acceleration factor
  - "max\_bias": The maximum value of the bias (If `null`, the script will estimate it as the maximum previous bias experienced.)
- "colvar\_name": The name of the colvar files.
- "log\_name": The name of the PLUMED log file used to determine if the simulation transitioned.
- "plog\_len": The line number of the COMMITTOR message which appears when the system transitions. The script assumes any PLUMED log longer than `plog_len` corresponds to a simulation that transitioned.
- "directories": The names of the directories for each simulation set to be analyzed.
- "num\_runs": The number of runs.
- "results\_file": The JSON file to save the results of the analysis to.

`Software/rate_analysis/rates.py` is the main Python script which runs the analysis. It is run with `python rates.py params.json`. The rates.py script assumes that each simulation set directory contains num_runs directories named run_#, each containing a colvar file and PLUMED log file named according to params.json. This script also hs a few anaysis-specific parameters:

- "analyses": A dictionary of rate methods to be used.
- "logTrick": Whether to use the *log-sum-exp* trick to deal with large exponents.
- "cores": The number of CPU cores to use.
- "ks\_ranges": Whether to calculate the range of $\gamma$ values and the coresponding range of best-fit $k\_0$ values which pass the KS test.
- "boots": Whether to perform bootstrapping for error bars. The error bars provided are the standard deviations in $\log_{10}k\_0$.
- "gamma\_bounds": The range of $\gamma$ values allowed during likelihood maximization and least-squares fitting.
- "runs": The names of the directories containing the colvar files and PLUMED logs.

The rates.py script is responsible calling the main function from rate\_methods.py to calculate the rate for each simulation set, and to save the results to the JSON file named in params.json.

`Software/rate_analysis/rate_methods.py` contains all the Python functions necessary to perform the rate analyses on a single simulation set. The main function, `rates()`, takes as parameters: directory (the path to the directory containing the simulation set), runs, analyses, columns, beta, gamma\_bounds, colvar\_name, log\_name, plog\_len, cores, ks\_ranges (default: False), boots (default: False), and logTrick (default: False). It returns a dictionary of results with keys of the form "\<method\> \<fit\> \<value\>", where:

- \<method\> can be one of "iMetaD", "KTR Vmb", or "EATR".
- \<fit\> can be one of "MLE" (likeihood maximization) or "CDF" (least-squares fitting)
- \<value\> can be one of "k" (rate), "std k" (standard error in rate), "KS klo" (lowest rate that can pass the KS test), "KS khi" (highest rate that can pass the KS test), "g" ($\gamma$, where applicable), "std g", "KS glo", or "KS ghi".

There are also several functions in "rate\_methods.py" which can run part of each rate method separately.
