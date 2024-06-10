#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12GB
#SBATCH --time=03:00:00
#SBATCH --job-name=Q_rates
#SBATCH --output=Q_slope.out


python ../../Software/rate_analysis/Q-OPES-slope.py


