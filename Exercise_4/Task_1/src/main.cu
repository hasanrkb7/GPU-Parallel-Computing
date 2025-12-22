
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

#define BS 16

__global__
void stencil2d_global(const double* x, double* y, int n, int m, double mu) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j < m) {
        int idx = i * m + j;
        double yy = (4.0 + mu) * x[idx];

        if (i > 0)     yy -= x[(i - 1) * m + j];
        if (i < n - 1) yy -= x[(i + 1) * m + j];
        if (j > 0)     yy -= x[i * m + (j - 1)];
        if (j < m - 1) yy -= x[i * m + (j + 1)];

        y[idx] = yy;
    }
}

__global__
void stencil2d_shared(const double* x, double* y, int n, int m, double mu) {
    __shared__ double tile[BS+2][BS+2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int j = blockIdx.x * BS + tx;
    int i = blockIdx.y * BS + ty;

    int tj = tx + 1;
    int ti = ty + 1;

    tile[ti][tj] = (i < n && j < m) ? x[i*m + j] : 0.0;

    if (tx == 0)
        tile[ti][0] = (j > 0 && i < n) ? x[i*m + (j-1)] : 0.0;
    if (tx == BS-1)
        tile[ti][tj+1] = (j < m-1 && i < n) ? x[i*m + (j+1)] : 0.0;
    if (ty == 0)
        tile[0][tj] = (i > 0 && j < m) ? x[(i-1)*m + j] : 0.0;
    if (ty == BS-1)
        tile[ti+1][tj] = (i < n-1 && j < m) ? x[(i+1)*m + j] : 0.0;

    __syncthreads();

    if (i < n && j < m) {
        double yy = (4.0 + mu) * tile[ti][tj]
                  - tile[ti-1][tj] - tile[ti+1][tj]
                  - tile[ti][tj-1] - tile[ti][tj+1];
        y[i*m + j] = yy;
    }
}

void cpu_ref(const double* x, double* y, int n, int m, double mu) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j) {
            int idx = i * m + j;
            double yy = (4.0 + mu) * x[idx];
            if (i > 0)     yy -= x[(i - 1) * m + j];
            if (i < n - 1) yy -= x[(i + 1) * m + j];
            if (j > 0)     yy -= x[i * m + (j - 1)];
            if (j < m - 1) yy -= x[i * m + (j + 1)];
            y[idx] = yy;
        }
}

float time_global(const double* dx, double* dy, int n, int m, double mu,
                  dim3 grid, dim3 block) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    stencil2d_global<<<grid, block>>>(dx, dy, n, m, mu);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return ms;
}

float time_shared(const double* dx, double* dy, int n, int m, double mu,
                  dim3 grid, dim3 block) {
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);
    cudaEventRecord(s);
    stencil2d_shared<<<grid, block>>>(dx, dy, n, m, mu);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);
    cudaEventDestroy(s);
    cudaEventDestroy(e);
    return ms;
}

int main() {
    srand(time(nullptr));

    int n = 500 + rand() % 1500;
    int m = 500 + rand() % 1500;
    double mu = 5.0 * rand() / RAND_MAX;
    if (mu == 0.0) mu = 0.1;

    size_t bytes = n * m * sizeof(double);

    double *hx = (double*)malloc(bytes);
    double *hy = (double*)malloc(bytes);
    double *href = (double*)malloc(bytes);

    for (int i = 0; i < n*m; ++i)
        hx[i] = 2.0 * rand() / RAND_MAX - 1.0;

    double *dx, *dy;
    cudaMalloc(&dx, bytes);
    cudaMalloc(&dy, bytes);
    cudaMemcpy(dx, hx, bytes, cudaMemcpyHostToDevice);

    dim3 block(BS, BS);
    dim3 grid((m + BS - 1) / BS, (n + BS - 1) / BS);

    float tg = time_global(dx, dy, n, m, mu, grid, block);
    cudaMemcpy(hy, dy, bytes, cudaMemcpyDeviceToHost);
    cpu_ref(hx, href, n, m, mu);

    double err = 0.0;
    for (int i = 0; i < n*m; ++i)
        err = fmax(err, fabs(hy[i] - href[i]));

    float ts = time_shared(dx, dy, n, m, mu, grid, block);

    printf("n=%d m=%d mu=%.3f\n", n, m, mu);
    printf("Max error: %.3e\n", err);
    printf("Global kernel time: %.3f ms\n", tg);
    printf("Shared kernel time: %.3f ms\n", ts);

    cudaFree(dx);
    cudaFree(dy);
    free(hx);
    free(hy);
    free(href);
    return 0;
}
