# Exercise 5 / Task 1 - Conjugate Gradient (CUDA)
# Build on Puhti (CSC) or any system with CUDA toolkit + nvcc

NVCC        ?= nvcc
TARGET      ?= cg
SRC         := src/main.cu

# Build for common CSC GPU architectures (V100 sm_70, A100 sm_80).
# If your system has a different GPU, you can override NVCCFLAGS_ARCH.
NVCCFLAGS_ARCH ?= -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80

NVCCFLAGS   ?= -O3 -std=c++17 -lineinfo $(NVCCFLAGS_ARCH)

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	rm -f $(TARGET) *.o
