# Exercise 2 / Task 1 - CUDA AXPY
# Build on Linux (e.g., Puhti) with: make
# You can override NVCC and CUDAFLAGS if needed:
#   make NVCC=nvcc CUDAFLAGS="-O3 -std=c++17"

NVCC ?= nvcc
CUDAFLAGS ?= -O3 -std=c++17

TARGET = axpy
SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CUDAFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)
