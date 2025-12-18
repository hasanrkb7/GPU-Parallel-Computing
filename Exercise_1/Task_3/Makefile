CXX = g++
CXXFLAGS = -O2 -std=c++17 -Wall -Wextra -pedantic

TARGET = dotprod
OBJS = main.o dot.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS)

main.o: main.cpp dot.h
	$(CXX) $(CXXFLAGS) -c main.cpp

dot.o: dot.cpp dot.h
	$(CXX) $(CXXFLAGS) -c dot.cpp

clean:
	rm -f $(TARGET) $(OBJS)
