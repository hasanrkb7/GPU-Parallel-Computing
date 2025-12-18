#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include "dot.h"

static double rand_double(double A, double B) {
    return (B - A) * (static_cast<double>(std::rand()) / RAND_MAX) + A;
}

int main() {
    std::srand(static_cast<unsigned>(std::time(nullptr)));

    // Random vector length in [10, 100000]
    int n = 10 + (std::rand() % (100000 - 10 + 1));

    std::vector<double> x(n), y(n);

    // Random values in [-1, 1]
    for (int i = 0; i < n; ++i) {
        x[i] = rand_double(-1.0, 1.0);
        y[i] = rand_double(-1.0, 1.0);
    }

    double d = dot(x.data(), y.data(), n);

    std::cout << "n = " << n << "\n";
    std::cout << "dot(x,y) = " << d << "\n";
    return 0;
}
