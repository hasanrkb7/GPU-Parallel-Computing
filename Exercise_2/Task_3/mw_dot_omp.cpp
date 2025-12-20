#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

// Matrix-weighted dot product:
// dot = sum_{i=0..n-1} sum_{j=0..n-1} A[i][j] * x[i] * y[j]
//
// This file contains:
//  - serial reference implementation
//  - OpenMP parallel implementation (collapse + reduction)
//  - correctness check
//  - strong scalability benchmark and CSV output

using Clock = std::chrono::high_resolution_clock;

static double now_seconds() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

static double urand(double lo, double hi) {
    const double r = (double)std::rand() / (double)RAND_MAX;
    return lo + (hi - lo) * r;
}

static double mw_dot_serial(const std::vector<double>& A,
                            const std::vector<double>& x,
                            const std::vector<double>& y,
                            int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        const double xi = x[i];
        const int row = i * n;
        for (int j = 0; j < n; j++) {
            sum += A[row + j] * xi * y[j];
        }
    }
    return sum;
}

static double mw_dot_omp(const std::vector<double>& A,
                         const std::vector<double>& x,
                         const std::vector<double>& y,
                         int n) {
    double sum = 0.0;

    // Parallelize both loops using collapse(2). Each iteration is independent.
    // Reduction is the correct way to safely accumulate to "sum".
    #pragma omp parallel for collapse(2) reduction(+:sum) schedule(static)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum += A[i * n + j] * x[i] * y[j];
        }
    }
    return sum;
}

static void write_csv_header(std::ofstream& out) {
    out << "threads,time_seconds,speedup,efficiency,checksum\n";
}

int main(int argc, char** argv) {
    // Usage:
    //   ./mw_dot_omp [n] [repeats] [csv_out]
    //
    // Defaults:
    //   n = 2000  (O(10^3), large enough for meaningful timing)
    //   repeats = 5
    //   csv_out = scalability_task3.csv

    int n = (argc >= 2) ? std::atoi(argv[1]) : 2000;
    int repeats = (argc >= 3) ? std::atoi(argv[2]) : 5;
    std::string csv_out = (argc >= 4) ? argv[3] : "scalability_task3.csv";

    if (n <= 0 || repeats <= 0) {
        std::cerr << "n and repeats must be > 0\n";
        return 1;
    }

    // Allocate data
    std::vector<double> A((size_t)n * (size_t)n);
    std::vector<double> x(n), y(n);

    // Deterministic seed for reproducibility
    std::srand(12345);

    // Initialize
    for (int i = 0; i < n; i++) {
        x[i] = urand(-1.0, 1.0);
        y[i] = urand(-1.0, 1.0);
    }
    for (size_t k = 0; k < A.size(); k++) {
        A[k] = urand(-1.0, 1.0);
    }

    // Correctness check (one run)
    const double ref = mw_dot_serial(A, x, y, n);
    const double par = mw_dot_omp(A, x, y, n);
    const double abs_err = std::abs(ref - par);
    const double rel_err = abs_err / (std::abs(ref) + 1e-30);

    std::cout << std::setprecision(16);
    std::cout << "Task 3: Matrix-weighted dot product (OpenMP)\n";
    std::cout << "n: " << n << " (matrix is " << n << "x" << n << ")\n";
    std::cout << "serial: " << ref << "\n";
    std::cout << "omp:    " << par << "\n";
    std::cout << "abs_err: " << abs_err << "\n";
    std::cout << "rel_err: " << rel_err << "\n";

    if (rel_err > 1e-10) {
        std::cerr << "FAILED correctness check (rel_err > 1e-10)\n";
        return 2;
    }
    std::cout << "PASSED correctness check\n\n";

    // Strong scalability benchmark
    const int max_threads = std::max(1, omp_get_num_procs());
    std::vector<int> thread_list;

    // Use powers of two up to max_threads, and also include max_threads if not power of two.
    for (int t = 1; t <= max_threads; t *= 2) thread_list.push_back(t);
    if (thread_list.back() != max_threads) thread_list.push_back(max_threads);

    std::ofstream csv(csv_out);
    if (!csv) {
        std::cerr << "Could not open CSV output: " << csv_out << "\n";
        return 3;
    }
    write_csv_header(csv);

    // Baseline time at 1 thread
    omp_set_num_threads(1);
    double best1 = 1e100;
    double checksum1 = 0.0;
    for (int r = 0; r < repeats; r++) {
        const double t0 = now_seconds();
        const double s = mw_dot_omp(A, x, y, n);
        const double t1 = now_seconds();
        checksum1 += s * 1e-300; // keep it tiny, but depend on computation
        best1 = std::min(best1, (t1 - t0));
    }

    for (int t : thread_list) {
        omp_set_num_threads(t);

        double best = 1e100;
        double checksum = 0.0;

        // A small warm-up
        (void)mw_dot_omp(A, x, y, n);

        for (int r = 0; r < repeats; r++) {
            const double t0 = now_seconds();
            const double s = mw_dot_omp(A, x, y, n);
            const double t1 = now_seconds();
            checksum += s * 1e-300;
            best = std::min(best, (t1 - t0));
        }

        const double speedup = best1 / best;
        const double efficiency = speedup / (double)t;

        csv << t << "," << best << "," << speedup << "," << efficiency << "," << checksum << "\n";

        std::cout << "threads=" << std::setw(3) << t
                  << "  best_time=" << std::setw(10) << best << " s"
                  << "  speedup=" << std::setw(8) << speedup
                  << "  eff=" << std::setw(8) << efficiency
                  << "\n";
    }

    std::cout << "\nWrote CSV: " << csv_out << "\n";
    std::cout << "Next: plot using gnuplot (plot_task3.p) or python (plot_task3.py).\n";
    return 0;
}
