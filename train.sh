#!/bin/bash
#SBATCH --job-name training
<<<<<<< HEAD
#SBATCH --output=logs/all-fcn-musiccnn-%j.out                # Name of output file (%j expands to jobId)
=======
#SBATCH --output=scripts/transfer-%j.out                # Name of output file (%j expands to jobId)
>>>>>>> 071d5bb6c545136e89d87e5261fc6022a901ca31
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
<<<<<<< HEAD
python train.py --model_class_name="FCN3" --num_workers=16 --apply_transformations
python train.py --model_class_name="FCN4" --num_workers=16 --apply_transformations
python train.py --model_class_name="FCN5" --num_workers=16 --apply_transformations
python train.py --model_class_name="MusicCNN" --num_workers=16 --apply_transformations
=======

python train.py --model_class_name="FCN7Transfer1Layer" --apply_transformations --num_workers=16 --apply_transfer --train_annotations='gtzan_train_label.csv' --val_annotations='gtzan_val_label.csv' --test_annotations='gtzan_test_label.csv'

python train.py --model_class_name="FCN7Transfer2Layers" --apply_transformations --num_workers=16 --apply_transfer --train_annotations='gtzan_train_label.csv' --val_annotations='gtzan_val_label.csv' --test_annotations='gtzan_test_label.csv'

python train.py --model_class_name="FCN7TransferUnfreezed" --apply_transformations --num_workers=16 --apply_transfer --train_annotations='gtzan_train_label.csv' --val_annotations='gtzan_val_label.csv' --test_annotations='gtzan_test_label.csv'
>>>>>>> 071d5bb6c545136e89d87e5261fc6022a901ca31
