#!/bin/bash
#SBATCH --job-name training
#SBATCH --output=train.%j.out                # Name of output file (%j expands to jobId)
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16        
#SBATCH --mem=64G
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 16:00:00
#SBATCH --partition brown                   
#SBATCH --mail-type=BEGIN,END,FAIL               # When to send email
#SBATCH --mail-user=$USER@itu.dk            # Email to which to send updates

# load modules or conda environments here
# update the $CONDAENV here to match the environment created earlier
module load Anaconda3
source activate aml-project

# Run script
python train.py --apply_transformations=True --apply_augmentations=False
