
import os,sys
if len(sys.argv) != 2:
    sys.exit("USAGE: python rates.py [parameter file]")

import json
sys.path.append(os.path.abspath('.'))
import numpy as np
import random
import glob
import rate_methods as RM
from scipy import interpolate, optimize
from scipy.stats import ks_1samp, ks_2samp
from scipy.stats import gamma as gamma_func
import multiprocessing as mp
from functools import partial


with open(sys.argv[1],'r') as params_file:
    params = json.load(params_file)
locals().update(params)


# Which analyses would you like to perform?
analyses = {
        "iMetaD MLE": True, # Infrequent Metadynamics, k = 1 / <tau>
        "iMetaD CDF": True, # Infrequent Metadynamics, k obtained from fitting P(t) = 1 - e^(-k tau) (CDF fit refinement)
        "KTR Vmb MLE": True, # Kramers' Time-dependent Rate Method, original
        "KTR Vmb CDF": True, # Kramers' Time-dependent Rate Method, original + CDF fit refinement
        "EATR MLE": True, # Same procedure as KTR, but S(t) = e^(-k <int_0^t e^(beta gamma V(t')) dt'>)
        "EATR CDF": True # Same as above + CDF fit refinement
}

# Use log-sum-exp trick to increase precision for exponents of large bias energies
logTrick = False

# Number of cores on which to run integral calculations.
cores = 4

ks_ranges = False # Find range of k0s and gammas where the KS test passes.
boots = True # Calculate errors using bootstrap analysis. 

gamma_bounds = (1.,1.00000000001) # The boundaries for the bounded optimization of gamma.

# Names of directories for each run.
runs = [f"run_{i+1}" for i in range(num_runs)]
# Feel free to change the filename structure in rate_methods.py if it is not compatable with your data.

# Seed for random number generator. Use a=None to set seed to current system time.
random.seed(a=12345) 


results = {
        "iMetaD MLE k": [],
        "iMetaD MLE std k": [],
        "iMetaD CDF k": [],
        "iMetaD CDF std k": [],
        "iMetaD CDF KS klo": [],
        "iMetaD CDF KS khi": [],
        "KTR Vmb MLE k": [],
        "KTR Vmb MLE std k": [],
        "KTR Vmb MLE g": [],
        "KTR Vmb MLE std g": [],
        "KTR Vmb CDF k": [],
        "KTR Vmb CDF std k": [],
        "KTR Vmb CDF g": [],
        "KTR Vmb CDF std g": [],
        "KTR Vmb CDF KS klo": [],
        "KTR Vmb CDF KS khi": [],
        "KTR Vmb CDF KS glo": [],
        "KTR Vmb CDF KS ghi": [],
        "EATR MLE k": [],
        "EATR MLE std k": [],
        "EATR MLE g": [],
        "EATR MLE std g": [],
        "EATR CDF k": [],
        "EATR CDF std k": [],
        "EATR CDF g": [],
        "EATR CDF std g": [],
        "EATR CDF KS klo": [],
        "EATR CDF KS khi": [],
        "EATR CDF KS glo": [],
        "EATR CDF KS ghi": []
}


for directory in directories:
    wd = head_dir + directory
    outcome = RM.rates(wd,runs,analyses,columns,beta,gamma_bounds,colvar_name,log_name,plog_len,cores,ks_ranges=ks_ranges,boots=boots,logTrick=logTrick)
    for key, value in outcome.items():
        if value is not None:
            results[key].append(value)
    print(f'Set {directory} complete!')

with open(results_file, 'w') as f:
    json.dump(results, f)

