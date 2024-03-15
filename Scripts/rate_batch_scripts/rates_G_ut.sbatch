#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1GB
#SBATCH --time=48:00:00
#SBATCH --job-name=Gu_rates
#SBATCH --output=G_ut.out


python ../../Software/rate_analysis/rates.py ../../Software/rate_analysis/params/params_G_ut.json


