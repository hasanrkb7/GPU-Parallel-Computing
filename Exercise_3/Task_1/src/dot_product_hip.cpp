#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

/*
 * Exercise 3 — Task 1
 * GPU dot product in double precision using shared-memory reduction.
 *
 * Requirements addressed:
 *  - input vectors remain unchanged (read-only)
 *  - global memory reads for x and y
 *  - per-thread partial sums + intra-block reduction in shared memory
 *  - dot computed fully on GPU; only 8 bytes (one double) copied back
 */

#define HIP_CHECK(call)                                                     \
  do {                                                                      \
    hipError_t err = (call);                                                \
    if (err != hipSuccess) {                                                \
      std::fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,     \
                   hipGetErrorString(err));                                 \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

/*
 * Kernel 1: compute one partial dot product per block.
 * Each thread accumulates a grid-stride sum in registers, then we reduce within
 * the block using shared memory.
 */
__global__ void dot_partial(const double* __restrict__ a,
                            const double* __restrict__ b,
                            double* __restrict__ partial,
                            size_t n) {
  extern __shared__ double sdata[];  // shared memory buffer for reduction
  const unsigned int tid = threadIdx.x;
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int stride = gridDim.x * blockDim.x;

  // Per-thread accumulation in registers (keeps inputs unchanged)
  double sum = 0.0;
  for (size_t i = idx; i < n; i += stride) {
    sum += a[i] * b[i];
  }

  // Store into shared memory and reduce
  sdata[tid] = sum;
  __syncthreads();

  // Tree reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // One partial sum per block in global memory
  if (tid == 0) {
    partial[blockIdx.x] = sdata[0];
  }
}

/*
 * Kernel 2: reduce the partial sums to a single double.
 * We use one block and reduce in shared memory.
 */
__global__ void reduce_final(const double* __restrict__ partial,
                             double* __restrict__ out,
                             size_t n) {
  extern __shared__ double sdata[];
  const unsigned int tid = threadIdx.x;

  // Each thread sums a strided subset of the partial array
  double sum = 0.0;
  for (size_t i = tid; i < n; i += blockDim.x) {
    sum += partial[i];
  }

  sdata[tid] = sum;
  __syncthreads();

  // Reduce within the block
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    *out = sdata[0];
  }
}

/*
 * Host reference dot product.
 * Uses long double to reduce accumulation error in the CPU check.
 */
static double cpu_dot(const std::vector<double>& a, const std::vector<double>& b) {
  long double acc = 0.0L;
  for (size_t i = 0; i < a.size(); ++i) {
    acc += static_cast<long double>(a[i]) * static_cast<long double>(b[i]);
  }
  return static_cast<double>(acc);
}

int main(int argc, char** argv) {
  // Default vector length (can be overridden by argv[1])
  size_t n = 1u << 24;
  if (argc >= 2) {
    n = std::stoull(argv[1]);
  }

  // Generate deterministic test vectors (host)
  std::vector<double> h_a(n), h_b(n);
  for (size_t i = 0; i < n; ++i) {
    h_a[i] = std::sin(static_cast<double>(i)) * 0.001 + 1.0;
    h_b[i] = std::cos(static_cast<double>(i)) * 0.002 + 2.0;
  }

  // Device buffers
  double *d_a = nullptr, *d_b = nullptr, *d_partial = nullptr, *d_out = nullptr;
  HIP_CHECK(hipMalloc(&d_a, n * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_b, n * sizeof(double)));

  // Launch configuration: a reasonable number of blocks (scaled by SM count)
  int device = 0;
  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, device));

  const int block = 256;
  int blocks = prop.multiProcessorCount * 8;
  if (blocks < 1) {
    blocks = 1;
  }

  HIP_CHECK(hipMalloc(&d_partial, static_cast<size_t>(blocks) * sizeof(double)));
  HIP_CHECK(hipMalloc(&d_out, sizeof(double)));

  // Copy inputs to device (inputs remain unchanged afterwards)
  HIP_CHECK(hipMemcpy(d_a, h_a.data(), n * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_b, h_b.data(), n * sizeof(double), hipMemcpyHostToDevice));

  // Time only the GPU work (kernels)
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipEventRecord(start, 0));
  dot_partial<<<blocks, block, block * sizeof(double), 0>>>(d_a, d_b, d_partial, n);
  HIP_CHECK(hipGetLastError());

  // Reduce partial sums to a single double
  reduce_final<<<1, 256, 256 * sizeof(double), 0>>>(d_partial, d_out, static_cast<size_t>(blocks));
  HIP_CHECK(hipGetLastError());

  HIP_CHECK(hipEventRecord(stop, 0));
  HIP_CHECK(hipEventSynchronize(stop));

  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

  // Requirement: copy back ONLY 8 bytes (one double)
  double h_out = 0.0;
  HIP_CHECK(hipMemcpy(&h_out, d_out, sizeof(double), hipMemcpyDeviceToHost));

  // Host verification
  const double ref = cpu_dot(h_a, h_b);
  const double abs_err = std::abs(h_out - ref);
  const double rel_err = abs_err / (std::abs(ref) + 1e-30);

  std::cout << "n=" << n << "\n";
  std::cout << "gpu_dot=" << h_out << "\n";
  std::cout << "cpu_dot=" << ref << "\n";
  std::cout << "abs_err=" << abs_err << "\n";
  std::cout << "rel_err=" << rel_err << "\n";
  std::cout << "kernel_ms=" << ms << "\n";

  // Cleanup
  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_a));
  HIP_CHECK(hipFree(d_b));
  HIP_CHECK(hipFree(d_partial));
  HIP_CHECK(hipFree(d_out));

  return 0;
}
