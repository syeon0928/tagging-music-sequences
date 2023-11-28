#!/bin/bash
#SBATCH --job-name training
#SBATCH --output=scripts/model-%j.out                # Name of output file (%j expands to jobId)
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16        
#SBATCH --mem=16G
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 16:00:00
#SBATCH --partition brown                   
#SBATCH --mail-type=BEGIN,END,FAIL               # When to send email
#SBATCH --mail-user=abys@itu.dk           # Email to which to send updates

# load modules or conda environments here
# update the $CONDAENV here to match the environment created earlier
module load Anaconda3
source activate aml-project

# Run script
python train.py --model_class_name="" --shuffle_train --apply_transformations --num_workers=16
