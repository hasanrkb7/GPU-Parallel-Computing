#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            std::cerr << "CUDA error: " << cudaGetErrorString(err)           \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// -------------------------- kernels --------------------------

__global__ void set_zero(double* a, std::size_t n) {
    std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0.0;
}

// y = A*x for 2D 5-point stencil on (n x m) logical grid with padded row length mbar.
// Flattened index: idx = i*mbar + j, with 0<=j<mbar, but only 0<=j<m are "real" elements.
__global__ void mulA_stencil(double* __restrict__ y,
                             const double* __restrict__ x,
                             int n, int m, int mbar, double mu) {
    std::size_t tid = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    std::size_t bn = (std::size_t)n * (std::size_t)mbar;
    if (tid >= bn) return;

    int i = (int)(tid / (std::size_t)mbar);
    int j = (int)(tid - (std::size_t)i * (std::size_t)mbar);

    // padded region -> keep as zero so dot products over bn are safe
    if (j >= m) {
        y[tid] = 0.0;
        return;
    }

    double center = x[tid];
    double up    = (i > 0)     ? x[(std::size_t)(i - 1) * mbar + j] : 0.0;
    double down  = (i + 1 < n) ? x[(std::size_t)(i + 1) * mbar + j] : 0.0;
    double left  = (j > 0)     ? x[(std::size_t)i * mbar + (j - 1)] : 0.0;
    double right = (j + 1 < m) ? x[(std::size_t)i * mbar + (j + 1)] : 0.0;

    y[tid] = (mu + 4.0) * center - up - down - left - right;
}

// y = y + a*x   (AXPY)
__global__ void axpy(double* __restrict__ y, const double* __restrict__ x, double a, std::size_t n) {
    std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

// y = y - a*x  (AXPY variant)
__global__ void axpy_sub(double* __restrict__ y, const double* __restrict__ x, double a, std::size_t n) {
    std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    if (i < n) y[i] -= a * x[i];
}

// p = beta*p + r  (modified AXPY)
__global__ void axpby(double* __restrict__ p, const double* __restrict__ r, double beta, std::size_t n) {
    std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    if (i < n) p[i] = beta * p[i] + r[i];
}

// r = b - Ax
__global__ void init_r(double* __restrict__ r, const double* __restrict__ b, const double* __restrict__ Ax, std::size_t n) {
    std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;
    if (i < n) r[i] = b[i] - Ax[i];
}

// dot product using block reduction + atomic add
__global__ void dot_partial(const double* __restrict__ a, const double* __restrict__ b, std::size_t n, double* out) {
    extern __shared__ double sdata[];
    std::size_t tid = threadIdx.x;
    std::size_t i = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x;

    double sum = 0.0;
    // grid-stride loop
    std::size_t stride = (std::size_t)blockDim.x * (std::size_t)gridDim.x;
    while (i < n) {
        sum += a[i] * b[i];
        i += stride;
    }
    sdata[tid] = sum;
    __syncthreads();

    // reduction
    for (std::size_t s = (std::size_t)blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, sdata[0]);
}

// -------------------------- helpers --------------------------

static int get_arg_int(int argc, char** argv, const std::string& key, int def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) return std::atoi(argv[i + 1]);
    }
    return def;
}

static double get_arg_double(int argc, char** argv, const std::string& key, double def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == key) return std::atof(argv[i + 1]);
    }
    return def;
}

static void print_usage() {
    std::cout << "Usage: ./cg [--n N] [--m M] [--mu MU] [--eps EPS] [--max MAX]\n";
}

// -------------------------- main --------------------------

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--help") {
        print_usage();
        return 0;
    }

    const int n = get_arg_int(argc, argv, "--n", 1000);
    const int m = get_arg_int(argc, argv, "--m", 1000);
    const double mu = get_arg_double(argc, argv, "--mu", 10.0);
    const double eps = get_arg_double(argc, argv, "--eps", 1e-8);
    const int maxit = get_arg_int(argc, argv, "--max", 30);

    if (n <= 0 || m <= 0 || maxit <= 0) {
        std::cerr << "Invalid parameters.\n";
        return 1;
    }
    if (mu <= 0.0) {
        std::cerr << "mu must be > 0 for SPD.\n";
        return 1;
    }

    // pad row length to multiple of 32
    const int mbar = ((m + 31) / 32) * 32;
    const std::size_t bn = (std::size_t)n * (std::size_t)mbar;

    std::cout << "n=" << n << " m=" << m << " mbar=" << mbar << " bn=" << bn
              << " mu=" << mu << " eps=" << eps << " max=" << maxit << "\n";

    // Host init
    std::vector<double> h_x(bn, 0.0);
    std::vector<double> h_b(bn, 0.0);

    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    // Fill only logical m columns, keep padding = 0
    for (int i = 0; i < n; ++i) {
        std::size_t row = (std::size_t)i * (std::size_t)mbar;
        for (int j = 0; j < m; ++j) {
            h_b[row + (std::size_t)j] = dist(rng);
            // initial guess x0 = 0 by default (could randomize if wanted)
        }
    }

    // Device alloc
    double *d_x=nullptr, *d_b=nullptr, *d_r=nullptr, *d_p=nullptr, *d_w=nullptr, *d_tmp=nullptr;
    double *d_dot=nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, bn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_b, bn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_r, bn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p, bn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_w, bn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tmp, bn * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dot, sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), bn * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), bn * sizeof(double), cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = (int)((bn + block - 1) / block);

    // r = b - A*x
    mulA_stencil<<<grid, block>>>(d_tmp, d_x, n, m, mbar, mu);
    CUDA_CHECK(cudaGetLastError());
    init_r<<<grid, block>>>(d_r, d_b, d_tmp, bn);
    CUDA_CHECK(cudaGetLastError());

    // p = r
    CUDA_CHECK(cudaMemcpy(d_p, d_r, bn * sizeof(double), cudaMemcpyDeviceToDevice));

    auto dot = [&](const double* a, const double* b)->double {
        CUDA_CHECK(cudaMemset(d_dot, 0, sizeof(double)));
        int grid_dot =  min(1024, (int)((bn + block - 1) / block));
        std::size_t shmem = (std::size_t)block * sizeof(double);
        dot_partial<<<grid_dot, block, shmem>>>(a, b, bn, d_dot);
        CUDA_CHECK(cudaGetLastError());
        double h=0.0;
        CUDA_CHECK(cudaMemcpy(&h, d_dot, sizeof(double), cudaMemcpyDeviceToHost));
        return h;
    };

    double lh = dot(d_r, d_r);  // lh = r·r
    double sqrt_h = std::sqrt(lh);

    std::cout << "iter 0: sqrt(h)=" << sqrt_h << "\n";
    if (sqrt_h < eps) {
        std::cout << "Converged at iter 0.\n";
        return 0;
    }

    // Timing
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    int it = 0;
    for (it = 1; it <= maxit; ++it) {
        // w = A*p
        mulA_stencil<<<grid, block>>>(d_w, d_p, n, m, mbar, mu);
        CUDA_CHECK(cudaGetLastError());

        // alpha = lh / dot(p, w)
        double pw = dot(d_p, d_w);
        double alpha = lh / pw;

        // x = x + alpha*p
        axpy<<<grid, block>>>(d_x, d_p, alpha, bn);
        CUDA_CHECK(cudaGetLastError());

        // r = r - alpha*w
        axpy_sub<<<grid, block>>>(d_r, d_w, alpha, bn);
        CUDA_CHECK(cudaGetLastError());

        // h = dot(r, r)
        double h = dot(d_r, d_r);
        sqrt_h = std::sqrt(h);

        std::cout << "iter " << it << ": sqrt(h)=" << sqrt_h << "\n";

        if (sqrt_h < eps) {
            lh = h;
            break;
        }

        // beta = h / lh
        double beta = h / lh;

        // p = beta*p + r
        axpby<<<grid, block>>>(d_p, d_r, beta, bn);
        CUDA_CHECK(cudaGetLastError());

        lh = h;
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev_start, ev_stop));

    std::cout << "Finished at iter " << it << " (max " << maxit << "), last sqrt(h)=" << sqrt_h << "\n";
    std::cout << "GPU CG loop time (events): " << ms << " ms\n";

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_r));
    CUDA_CHECK(cudaFree(d_p));
    CUDA_CHECK(cudaFree(d_w));
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_dot));

    return 0;
}
