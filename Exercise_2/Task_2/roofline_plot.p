# Usage: gnuplot roofline_plot.p
# Produces: roofline_plot.pdf

set terminal pdf size 10cm,8cm
set output 'roofline_plot.pdf'

set grid
set logscale xy

set title "Roofline model (Tesla K40c) — Exercise 2 Task 2"
set xlabel "Operational intensity (FLOP/byte)"
set ylabel "Performance (GFLOP/s)"

# Ranges (adjust if needed)
set xrange [0.01:1000]
set yrange [1:5000]

# ---- Parameters (edit these) ----
P_peak = 1430.0          # GFLOP/s
BW_nom = 288.0           # GB/s
BW_ecc = 0.8 * BW_nom    # simple ECC adjustment

I_kernel = 1.1875        # FLOP/byte (edit based on your count)
P_meas  = 196.866        # GFLOP/s

# Rooflines
Pmem_nom(x) = BW_nom * x
Pmem_ecc(x) = BW_ecc * x
Pcomp(x)    = P_peak

# Mark kernel point
set object circle at first I_kernel, P_meas radius char 0.5 fillstyle empty border lw 2

plot \
    Pcomp(x)    lw 3 title "Compute peak", \
    Pmem_nom(x) lw 3 title "Memory roof (288 GB/s)", \
    Pmem_ecc(x) lw 3 title "Memory roof (ECC-adjusted)", \
    P_meas      lw 1 notitle
