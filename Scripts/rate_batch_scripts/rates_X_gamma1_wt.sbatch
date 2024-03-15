#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1GB
#SBATCH --time=48:00:00
#SBATCH --job-name=Xw1_rates
#SBATCH --output=X_wt.out


python ../../Software/rate_analysis/rates_gamma1.py ../../Software/rate_analysis/params/params_X_gamma1_wt.json


