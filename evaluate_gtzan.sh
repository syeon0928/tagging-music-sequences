#!/bin/bash
#SBATCH --job-name evaluation
#SBATCH --output=logs/fcn7transferunfreezed-evaluation-%j.out                # Name of output file (%j expands to jobId)
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

# Run script
python evaluate.py --test_annotations "gtzan_test_label.csv" --model_path "models/FCN7TransferUnfreezed_best.pth" --model_class_name "FCN7TransferUnfreezed"

python evaluate.py --test_annotations "gtzan_test_label.csv" --model_path "models/FCN7Transfer2layers_best.pth" --model_class_name "FCN7Transfer2Layers"

python evaluate.py --test_annotations "gtzan_test_label.csv" --model_path "models/FCN7Transfer1layer_best.pth" --model_class_name "FCN7Transfer1Layer"

