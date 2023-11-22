#!/bin/bash
#SBATCH --job-name train-model
#SBATCH --output=train-model.%j.out                # Name of output file (%j expands to jobId)
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16        
#SBATCH --mem=64G
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 8:00:00
#SBATCH --partition brown                   
#SBATCH --mail-type=BEGIN,END,FAIL               # When to send email
#SBATCH --mail-user=yuzh@itu.dk            # Email to which to send updates

# load modules or conda environments here
# update the $CONDAENV here to match the environment created earlier
module load Anaconda3
cd ../Tagging-Music_Sequences/notebooks
# Run script
jupyter notebook crnn_spec_yc.ipynb
