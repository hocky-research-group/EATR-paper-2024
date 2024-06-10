#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --time=03:00:00
#SBATCH --job-name=E_rates
#SBATCH --output=E_slope.out


python ../../Software/rate_analysis/E-OPES-slope.py


