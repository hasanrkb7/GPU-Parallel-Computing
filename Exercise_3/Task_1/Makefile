# Exercise 3 / Task 1 — Makefile (Puhti HIP)
# Usage:
#   make            # builds ./dot
#   make run        # runs with N=16777216 (can override: make run N=1000000)
#   make clean

CXX      ?= hipcc
CXXFLAGS ?= -O3 -std=c++17

TARGET   := dot
SRC      := src/dot_product_hip.cpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@

run: $(TARGET)
	./$(TARGET) $(N)

clean:
	rm -f $(TARGET) *.o
