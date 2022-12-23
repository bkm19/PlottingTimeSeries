#!/bin/bash

#SBATCH --job-name=time_series_boxplots #Job name
#SBATCH --ntasks=1 # Run on n CPU
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G #Job memory request
#SBATCH --time=48:00:00 # Time limit hrs:min:sec
#SBATCH --output=./logs/%x_%j.log # Standard output and error log

python3 allOneIterationResults.py
