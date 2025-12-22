# Exercise 5 – Task 1: Conjugate Gradient (CUDA)

CUDA implementation of the Conjugate Gradient method using double precision.
Matrix-free SPD 2D 5-point stencil mulA with padded rows (bn ≈ 1e6).

## Verified Puhti Run
- n = 1000, m = 1000, mbar = 1024 (bn = 1,024,000)
- μ = 10, eps = 1e-8, max = 30
- Converged in 13 iterations
- GPU CG loop time ≈ 2.5 ms (CUDA events)

## Files
- src/main.cu
- Makefile
- run_puhti.sh
