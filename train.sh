#!/bin/bash
#SBATCH --job-name training
#SBATCH --output=logs/all-fcn-musiccnn-%j.out                # Name of output file (%j expands to jobId)
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16        
#SBATCH --mem=32G
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 16:00:00
#SBATCH --partition brown                   
#SBATCH --mail-type=BEGIN,END,FAIL               # When to send email
#SBATCH --mail-user=abys@itu.dk           # Email to which to send updates

# Activate environment
module load Anaconda3
source activate aml-project

# Run script
python train.py --model_class_name="FCN3" --num_workers=16 --apply_transformations
python train.py --model_class_name="FCN4" --num_workers=16 --apply_transformations
python train.py --model_class_name="FCN5" --num_workers=16 --apply_transformations
python train.py --model_class_name="MusicCNN" --num_workers=16 --apply_transformations
