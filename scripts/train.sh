#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=./logs/%J_stdout.txt
#SBATCH --error=./logs/%J_stderr.txt
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=newtonnet
#SBATCH --exclude=c302,c301

module load Mamba
module load CUDA/11.8.0
module load GCC/9.3.0
source ~/.bashrc
conda activate newtonnet


python newtonnet_train.py --config config.yml
