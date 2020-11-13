#include <omp.h>
#include <math.h>

#include "ekblas.h"

// vector dot product
float ek_sdot(const size_t n, const float *x, const size_t inc_x, const float *y, const size_t inc_y) {
    size_t i;
    float result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < n; i++) {
        result += x[i * inc_x] * y[i * inc_y];
    }
    return result;
}

// sum of the absolute values
float ek_sasum(const size_t n, const float *x, const size_t inc_x) {
    size_t i;
    float result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < n; i++) {
        result += fabsf(x[i * inc_x]);
    }
    return result;
}

// constant times a vector plus a vector
void ek_saxpy(const size_t n, const float alpha, const float *x, const size_t inc_x, float *y, const size_t inc_y) {
    size_t i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        y[i * inc_y] = alpha * x[i * inc_x] + y[i * inc_y];
    }
}

// euclidean norm of a vector
float ek_snrm2(const size_t n, const float *x, const size_t inc_x) {
    size_t i;
    float result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < n; i++) {
        result += x[i * inc_x] * x[i * inc_x];
    }
    return sqrt(result);
}