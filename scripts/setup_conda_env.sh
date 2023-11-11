#!/bin/bash

#SBATCH --job-name=pytorch-gpu-condaenv    # Job name
#SBATCH --output=job.%j.out                # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=2                  # Number of CPU cores per task
#SBATCH --time=08:00:00                    # Run time (hh:mm:ss) - 1 hour max
#SBATCH --gres=gpu                         # Request GPU resource
#SBATCH --partition=brown                  # Specify a partition
#SBATCH --mail-type=END,FAIL               # When to send email
#SBATCH --mail-user=abys@itu.dk            # Email to which to send updates

echo "Running on $(hostname):"
module load Anaconda3
conda create --name aml-project
source activate aml-project

conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -y jupyter ipykernel
python -m ipykernel install --user --name aml-project --display-name "Python (aml-project)"

# Verify install
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Install project requirements
pip install -r ../requirements.txt
