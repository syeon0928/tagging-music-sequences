#!/bin/bash
#SBATCH --job-name evaluation
#SBATCH --output=logs/evaluation-results-%j.out                # Name of output file (%j expands to jobId)
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=16        
#SBATCH --mem=32G
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 24:00:00
#SBATCH --partition brown                   
#SBATCH --mail-type=BEGIN,END,FAIL               # When to send email
#SBATCH --mail-user=abys@itu.dk           # Email to which to send updates

# Activate environment
module load Anaconda3
source activate aml-project

CLASS_NAME_REGEX='WaveCNN[0-9]+WithSelfAttention|FCN[0-9]+WithSelfAttention|WaveCNN[0-9]+|FCN[0-9]+|MusicCNN'

# Run script
for MODEL_PATH in models/*best*.pth; do
    # Extract the model class name from the filename
    FILENAME=$(basename "$MODEL_PATH")
    MODEL_CLASS_NAME=$(echo $FILENAME | grep -o -E "CLASS_NAME_REGEX")

    if [ -z "$MODEL_CLASS_NAME" ]; then
        echo "Could not determine model class for $MODEL_PATH. Skipping."
        continue
    fi

    echo "Evaluating model: $MODEL_PATH using class $MODEL_CLASS_NAME"
    python evaluate.py --model_path "$MODEL_PATH" --model_class_name "$MODEL_CLASS_NAME"
done
