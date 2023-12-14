#!/bin/bash
#SBATCH --job-name evaluation
#SBATCH --output=logs/MODELNAME-evaluation-%j.out                # Name of output file (%j expands to jobId)
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16        
#SBATCH --mem=32G
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 8:00:00
#SBATCH --partition brown                   
#SBATCH --mail-type=BEGIN,END,FAIL               # When to send email
#SBATCH --mail-user=abys@itu.dk           # Email to which to send updates

# Activate environment
module load Anaconda3
source activate aml-project

# Set params
TEST_ANNOTATIONS="gtzan_test_label.csv"
MODEL_PATH="models/FCN7TransferUnfreezed_best.pth"
MODEL_CLASS_NAME="FCN7TransferUnfreezed"

# Run script
python evaluate.py --test_annotations "$TEST_ANNOTATIONS" --model_path "$MODEL_PATH" --model_class_name "$MODEL_CLASS_NAME"

