#include <iostream>
#include <cuda_runtime.h>

int main() {
    std::cout
        << "The first task of the first exercises: / CUDA-example"
        << std::endl;

    std::cout
        << "Name of the person returning the exercise: Rakibul Hasan Remon"
        << std::endl;

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    std::cout << "Found " << deviceCount << " CUDA-devices:" << std::endl;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout
            << " " << prop.pciBusID << "/" << prop.pciDeviceID
            << ": " << prop.name
            << " (CC " << prop.major << "." << prop.minor << ")"
            << std::endl;
    }
    return 0;
}