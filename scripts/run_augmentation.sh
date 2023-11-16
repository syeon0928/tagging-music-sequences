#!/bin/bash
#SBATCH --job-name=augmentation  # Job name
#SBATCH --output=job.%j.out      # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=32       # Schedule one core
#SBATCH --time=08:00:00          # Run time (hh:mm:ss) - 8 hours max
#SBATCH --partition=brown    # Run on either the Red or Brown queue

module load Anaconda3
source activate aml-project
python ../src/audio_augmentor.py