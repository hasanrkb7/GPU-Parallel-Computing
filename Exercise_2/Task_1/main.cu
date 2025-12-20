#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

static void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in " << what << ": " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

__global__ void axpy_kernel(double* y, const double* x, double a, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < n; i += stride) {
        y[i] = a * x[i] + y[i];
    }
}

int main(int argc, char** argv) {
    // Usage: ./axpy [n] [a] [seed]
    int n = (argc >= 2) ? std::atoi(argv[1]) : (1 << 20); // default ~1M
    double a = (argc >= 3) ? std::atof(argv[2]) : 1.2345;
    unsigned seed = (argc >= 4) ? (unsigned)std::atoi(argv[3]) : 12345u;

    if (n <= 0) {
        std::cerr << "n must be > 0\n";
        return 1;
    }

    // Host data
    std::vector<double> x(n), y(n), y_ref(n);

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < n; i++) {
        x[i] = dist(rng);
        y[i] = dist(rng);
        y_ref[i] = a * x[i] + y[i]; // reference result
    }

    // Device data
    double *d_x = nullptr, *d_y = nullptr;
    checkCuda(cudaMalloc((void**)&d_x, n * sizeof(double)), "cudaMalloc(d_x)");
    checkCuda(cudaMalloc((void**)&d_y, n * sizeof(double)), "cudaMalloc(d_y)");

    checkCuda(cudaMemcpy(d_x, x.data(), n * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy H2D x");
    checkCuda(cudaMemcpy(d_y, y.data(), n * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy H2D y");

    // Kernel launch params
    int block = 256;
    int grid = (n + block - 1) / block;
    if (grid > 65535) grid = 65535; // simple cap for 1D grid on older limits

    axpy_kernel<<<grid, block>>>(d_y, d_x, a, n);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    checkCuda(cudaMemcpy(y.data(), d_y, n * sizeof(double), cudaMemcpyDeviceToHost), "cudaMemcpy D2H y");

    // Verify
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double err = std::abs(y[i] - y_ref[i]);
        if (err > max_err) max_err = err;
    }

    std::cout << std::setprecision(16);
    std::cout << "AXPY check\n";
    std::cout << "n: " << n << "\n";
    std::cout << "a: " << a << "\n";
    std::cout << "max_err: " << max_err << "\n";

    cudaFree(d_x);
    cudaFree(d_y);

    // A reasonable tolerance for double precision
    if (max_err > 1e-12) {
        std::cerr << "FAILED (max_err > 1e-12)\n";
        return 2;
    }
    std::cout << "PASSED\n";
    return 0;
}
