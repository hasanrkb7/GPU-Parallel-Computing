# Usage: gnuplot plot_task3.p
# Input:  scalability_task3.csv
# Output: scalability_task3.pdf

set terminal pdf size 10cm,8cm
set output 'scalability_task3.pdf'

set grid
set title "Exercise 2 Task 3 — Strong scalability (OpenMP)"
set xlabel "Threads"
set ylabel "Speedup (T1 / Tn)"

set key left top

# CSV
set datafile separator ","
# columns: 1=threads, 2=time_seconds, 3=speedup, 4=efficiency, 5=checksum

plot \
    'scalability_task3.csv' using 1:3 with linespoints title "Speedup", \
    x with lines title "Ideal"
