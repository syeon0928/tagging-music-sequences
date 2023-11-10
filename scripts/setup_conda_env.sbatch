#!/bin/bash

#SBATCH --job-name=pytorch-gpu-condaenv    # Job name
#SBATCH --output=job.%j.out                # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=2                  # Number of CPU cores per task
#SBATCH --time=01:00:00                    # Run time (hh:mm:ss) - 1 hour max
#SBATCH --gres=gpu                         # Request GPU resource
#SBATCH --partition=brown                  # Specify a partition
#SBATCH --mail-type=BEGIN,END,FAIL         # When to send email
#SBATCH --mail-user=abys@itu.dk            # Email to which to send updates

# Load Anaconda module
module load Anaconda3

# Create a new conda environment with required packages
conda create --name test_env

# Activate the environment
source activate test_env

# Install PyTorch with GPU support
conda install -y numpy

