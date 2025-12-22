NVCC = nvcc
CFLAGS = -O3

TARGET = stencil2d
SRC = src/main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CFLAGS) -o $@ $^

clean:
	rm -f $(TARGET)
