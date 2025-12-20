# Exercise 2 — Task 3 (OpenMP)

## What this task does
Implements the **matrix-weighted dot product** using OpenMP:

\[
\text{dot} = \sum_{i=0}^{n-1}\sum_{j=0}^{n-1} A_{i,j} x_i y_j
\]

Includes:
- serial reference result (correctness check)
- OpenMP parallel implementation
- **strong scalability** benchmark (vary thread count)
- writes `scalability_task3.csv`
- optional plotting scripts (gnuplot + python)

## Files
- `mw_dot_omp.cpp` — implementation + benchmark
- `Makefile` — builds with OpenMP
- `plot_task3.p` — gnuplot script (creates `scalability_task3.pdf`)
- `plot_task3.py` — python/matplotlib plot (creates `scalability_task3.png`)

## Build (Linux or MSYS2 MinGW)
```bash
make
```

## Run (example)
```bash
./mw_dot_omp 2000 5 scalability_task3.csv
```

Arguments:
- `n`        : matrix dimension (n x n). Recommended >= 1000.
- `repeats`  : how many times each configuration is timed (best time is taken).
- `csv_out`  : output CSV filename.

Expected output includes:
- "PASSED correctness check"
- timing lines per thread count
- "Wrote CSV: scalability_task3.csv"

## Plot (Option A: gnuplot)
```bash
gnuplot plot_task3.p
```
Creates: `scalability_task3.pdf`

## Plot (Option B: Python)
```bash
python3 plot_task3.py scalability_task3.csv
```
Creates: `scalability_task3.png`

## Notes on OpenMP settings
You can also force the number of threads:
```bash
export OMP_NUM_THREADS=8
```
But the program already benchmarks multiple thread counts automatically.
