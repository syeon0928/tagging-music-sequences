#!/bin/bash
#SBATCH --job-name bert
#SBATCH --output=scripts/att-wave-%j.out                # Name of output file (%j expands to jobId)
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16
#SBATCH --exclude=desktop[1-16,21,23-24]
#SBATCH --mem=64G
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 16:00:00
#SBATCH --partition brown                   
#SBATCH --mail-type=BEGIN,END,FAIL               # When to send email
#SBATCH --mail-user=yuzh@itu.dk           # Email to which to send updates

# load modules or conda environments here
# update the $CONDAENV here to match the environment created earlier

source activate aml-project

# Run script

python train.py --model_class_name 'WaveCNN7WithSelfAttention' --epochs 10 --batch_size 16

