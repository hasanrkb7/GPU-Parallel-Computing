#!/bin/bash
#SBATCH --job-name=stencil2d
#SBATCH --account=<YOUR_CSC_PROJECT>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=8G
#SBATCH --output=stencil2d.out

module purge
module load cuda

make clean
make

./stencil2d
