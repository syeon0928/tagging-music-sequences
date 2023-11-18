#!/bin/bash
#SBATCH --job-name jupyter-notebook
#SBATCH --output jupyter-notebook-%J.log
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=32        # Schedule 32 cores (idk what im doing)
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --time 08:00:00
#SBATCH --partition brown                   
# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

# print tunneling instructions jupyter-log
echo -e "
MacOS or linux terminal command to create your ssh tunnel
ssh -N -L ${port}:${node}:${port} ${user}@hpc.itu.dk

Windows MobaXterm info
Forwarded port: ${port}
Remote server: ${node}
Remote port: ${port}
SSH server: hpc.itu.dk
SSH login: $user
SSH port: 22

Use a Browser on your local machine to go to:
localhost:${port}  (prefix w/ https:// if using password)
"

# load modules or conda environments here
# update the $CONDAENV here to match the environment created earlier
module load Anaconda3
source activate aml-project

# Start the server
cd ../
jupyter-notebook --no-browser --port=${port} --ip=0.0.0.0
