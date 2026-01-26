# GPU Parallel Computing

These are my exercises from the GPU Programming course (TIES3420) at the University of Jyväskylä, from December 2025 to January 2026. I compiled and ran the GPU programs on the CSC Puhti supercomputer (NVIDIA V100 nodes) with Slurm batch jobs.

## What is inside

| Folder | What it does |
|---|---|
| `Exercise_0` | My first GPU program on Puhti, with the Slurm job script, the output log and a screenshot |
| `Exercise_1/Task_3` | A small C++ program split into separate files and built with a Makefile (dot product) |
| `Exercise_2/Task_1` | A CUDA AXPY kernel with a check on the CPU |
| `Exercise_2/Task_2` | Roofline model analysis of why a CUDA example reached about 197 GFlop/s and not the 1430 GFlop/s peak |
| `Exercise_2/Task_3` | Matrix-weighted dot product with OpenMP, with strong scaling measurements, a CSV file and a plot |
| `Exercise_2/Task_4` | A written analysis of a parallel code |
| `Exercise_3/Task_1` | Dot product in HIP with a shared-memory reduction |
| `Exercise_4/Task_1` | A 2D stencil in CUDA, with a shared-memory version |
| `Exercise_5/Task_1` | A Conjugate Gradient solver in CUDA (double precision, matrix-free 5-point stencil, around 1 million unknowns) |
| `Project` | Project proposal for K-means clustering with CUDA |

Every task folder has its own README with build and run notes.

## How to run on Puhti

Step 1: Put your own CSC project number in `#SBATCH --account=<YOUR_CSC_PROJECT>` in the job script.

Step 2: Load the compiler modules and build:

```bash
module load gcc/13.2.0
module load cuda/12.6.0
make
```

Step 3: Submit the job with `sbatch run_puhti.sh`.
