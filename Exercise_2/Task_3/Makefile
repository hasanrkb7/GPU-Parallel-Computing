# Exercise 2 / Task 3 - OpenMP matrix-weighted dot product
# Build: make
# Run:   ./mw_dot_omp [n] [repeats] [csv_out]

CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fopenmp

TARGET = mw_dot_omp
SRC = mw_dot_omp.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET) *.o scalability_task3.csv scalability_task3.pdf
