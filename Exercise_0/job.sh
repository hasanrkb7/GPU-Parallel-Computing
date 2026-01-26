#!/bin/bash
#SBATCH --job-name=exercise0
#SBATCH --account=<YOUR_CSC_PROJECT>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=4
#SBATCH --output=output.txt

module purge
module load gcc/13.2.0
module load cuda/12.6.0

nvidia-smi
./main
