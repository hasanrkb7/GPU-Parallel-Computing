#!/bin/bash
#SBATCH --job-name=ex5_cg
#SBATCH --account=<YOUR_CSC_PROJECT>
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --output=slurm-%j.out

# Puhti module environment (adjust versions if needed)
module purge
module load gcc/13.2.0
module load cuda/12.6.0

set -euo pipefail

echo "Building..."
make clean
make

echo "Running..."
# Default: n=1000, m=1000 -> padded bn ~ 1,000,000
./cg --n 1000 --m 1000 --mu 10 --eps 1e-8 --max 30
