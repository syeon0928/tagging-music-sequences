#!/bin/bash
#SBATCH --job-name train-model
#SBATCH --output=train-model.%j.out                # Name of output file (%j expands to jobId)
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=32        # Schedule 32 cores (idk what im doing)
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 8:00:00
#SBATCH --partition brown                   

# load modules or conda environments here
# update the $CONDAENV here to match the environment created earlier
module load Anaconda3
source activate aml-project

# Run script
python ../src/model_sy.py
