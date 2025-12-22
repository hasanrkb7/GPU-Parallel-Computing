# Exercise 3 — Task 1 (GPU dot product)

HIP implementation of dot product in double precision using shared-memory reduction.  
Only the final result (1 double = 8 bytes) is copied back to host.

## Files
- src/dot_product_hip.cpp
- puhti/run.sh

## Run on Puhti
1) Edit `puhti/run.sh` and set:
```bash
#SBATCH --account=<YOUR_CSC_PROJECT>
```

2) Submit:
```bash
cd Exercise_3/Task_1/puhti
sbatch run.sh
```

## Puhti result (job 31020696)
n=16777216  
gpu_dot=3.35544e+07  
cpu_dot=3.35544e+07  
abs_err=7.45058e-09  
rel_err=2.22045e-16  
kernel_ms=0.415424
