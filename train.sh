#!/bin/bash
#SBATCH --job-name training
#SBATCH --output=scripts/transfer-%j.out                # Name of output file (%j expands to jobId)
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16        
#SBATCH --mem=32G
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 24:00:00
#SBATCH --partition brown                   
#SBATCH --mail-type=BEGIN,END,FAIL               # When to send email
#SBATCH --mail-user=seuh@itu.dk           # Email to which to send updates

# Activate environment
module load Anaconda3
source activate aml-project

# Run script

python train.py --model_class_name="FCN7Transfer1Layer" --apply_transformations --num_workers=16 --apply_transfer --train_annotations='gtzan_train_label.csv' --val_annotations='gtzan_val_label.csv' --test_annotations='gtzan_test_label.csv'

python train.py --model_class_name="FCN7Transfer2Layers" --apply_transformations --num_workers=16 --apply_transfer --train_annotations='gtzan_train_label.csv' --val_annotations='gtzan_val_label.csv' --test_annotations='gtzan_test_label.csv'

python train.py --model_class_name="FCN7TransferUnfreezed" --apply_transformations --num_workers=16 --apply_transfer --train_annotations='gtzan_train_label.csv' --val_annotations='gtzan_val_label.csv' --test_annotations='gtzan_test_label.csv'
