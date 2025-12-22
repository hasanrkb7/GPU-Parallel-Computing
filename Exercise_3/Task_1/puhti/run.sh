#!/bin/bash
#SBATCH --job-name=ex3_t1_dot
#SBATCH --account=<YOUR_PROJECT>
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module purge
module load rocm

hipcc -O3 -std=c++17 ../src/dot_product_hip.cpp -o dot

# arg = N (vector length)
./dot 16777216
