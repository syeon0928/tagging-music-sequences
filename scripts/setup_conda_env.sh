#!/bin/bash

#SBATCH --job-name=env-setup    # Job name
#SBATCH --output=env-setup-%j.out                # Name of output file (%j expands to jobId)
#SBATCH --cpus-per-task=16                 # Number of CPU cores per task
#SBATCH --time=08:00:00                    # Run time (hh:mm:ss) - 1 hour max
#SBATCH --gres=gpu                         # Request GPU resource
#SBATCH --partition=brown                  # Specify a partition
#SBATCH --mail-type=END,FAIL               # When to send email
#SBATCH --mail-user=abys@itu.dk            # Email to which to send updates

echo "Running on $(hostname):"
echo "Beginning environment setup"
module load Anaconda3

# Check if the environment already exists
conda env list | grep aml-project &> /dev/null
if [ $? -ne 0 ]; then
    echo "Creating new conda environment: aml-project"
    conda create --name aml-project -y
else
    echo "Conda environment aml-project already exists. Activating and updating."
fi

# Activate environment
source activate aml-project

# Install hpc supported cuda version
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Setup jupyter kernel to work with conda
conda install -y jupyter ipykernel
python -m ipykernel install --user --name aml-project --display-name "Python (aml-project)"

# Verify cuda availability
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Install project requirements
pip install -r ../requirements.txt

echo "Environment setup completed"