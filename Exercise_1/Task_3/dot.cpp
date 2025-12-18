#include "dot.h"

double dot(const double* x, const double* y, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += x[i] * y[i];
    }
    return s;
}
