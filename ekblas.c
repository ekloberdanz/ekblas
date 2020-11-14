#include <omp.h>
#include <math.h>

#include "ekblas.h"

// Level 1 - single precision

// inner product of two vectors with extended precision accumulation
float ek_sdsdot(const size_t n, const float alpha, const float *x, const size_t inc_x, const float *y, const size_t inc_y) {
    size_t i;
    double acc = 0.0;
    float result = 0.0;
    #pragma omp parallel for reduction(+:acc)
    for (i = 0; i < n; i++) {
        acc += x[i * inc_x] * y[i * inc_y];
    }
    result = acc;
    result += alpha;
    return result;
}

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

// euclidean norm (L2 norm) of a vector
float ek_snrm2(const size_t n, const float *x, const size_t inc_x) {
    size_t i;
    float result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < n; i++) {
        result += x[i * inc_x] * x[i * inc_x];
    }
    return sqrt(result);
}

// scales a vector by a constant
void ek_sscal(const size_t n, const float alpha, float *x, const size_t inc_x) {
    size_t i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        x[i * inc_x] = alpha * x[i * inc_x];
    }
}

// interchanges two vectors
void ek_sswap(const size_t n, float *x, const size_t inc_x, float *y, const size_t inc_y) {
    size_t i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        float tmp = x[i * inc_x];
        x[i * inc_x] = y[i * inc_y];
        y[i * inc_y] = tmp;
    }
}

// copies a vector, x, to a vector, y
void ek_scopy(const size_t n, const float *x, const size_t inc_x, float *y, const size_t inc_y) {
    size_t i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        y[i * inc_y] = x[i * inc_x];
    }
}


// Level 1 - double precision

// inner product of two vectors with extended precision accumulation
double ek_dsdot(const size_t n, const float *x, const size_t inc_x, const float *y, const size_t inc_y) {
    size_t i;
    double acc = 0.0;
    float result = 0.0;
    #pragma omp parallel for reduction(+:acc)
    for (i = 0; i < n; i++) {
        acc += x[i * inc_x] * y[i * inc_y];
    }
    result = acc;
    return result;
}

// vector dot product
double ek_ddot(const size_t n, const double *x, const size_t inc_x, const double *y, const size_t inc_y) {
    size_t i;
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < n; i++) {
        result += x[i * inc_x] * y[i * inc_y];
    }
    return result;
}

// sum of the absolute values
double ek_dasum(const size_t n, const double *x, const size_t inc_x) {
    size_t i;
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < n; i++) {
        result += fabs(x[i * inc_x]);
    }
    return result;
}

// constant times a vector plus a vector
void ek_daxpy(const size_t n, const double alpha, const double *x, const size_t inc_x, double *y, const size_t inc_y) {
    size_t i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        y[i * inc_y] = alpha * x[i * inc_x] + y[i * inc_y];
    }
}

// euclidean norm (L2 norm) of a vector
double ek_dnrm2(const size_t n, const double *x, const size_t inc_x) {
    size_t i;
    double result = 0.0;
    #pragma omp parallel for reduction(+:result)
    for (i = 0; i < n; i++) {
        result += x[i * inc_x] * x[i * inc_x];
    }
    return sqrt(result);
}

// scales a vector by a constant
void ek_dscal(const size_t n, const double alpha, double *x, const size_t inc_x) {
    size_t i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        x[i * inc_x] = alpha * x[i * inc_x];
    }
}

// interchanges two vectors
void ek_dswap(const size_t n, double *x, const size_t inc_x, double *y, const size_t inc_y) {
    size_t i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        double tmp = x[i * inc_x];
        x[i * inc_x] = y[i * inc_y];
        y[i * inc_y] = tmp;
    }
}

// copies a vector, x, to a vector, y
void ek_dcopy(const size_t n, const double *x, const size_t inc_x, double *y, const size_t inc_y) {
    size_t i;
    #pragma omp parallel for
    for (i = 0; i < n; i++) {
        y[i * inc_y] = x[i * inc_x];
    }
}



// // performs one of the matrix-vector operations
// void ek_sgbmv(const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE TransA, const int M, const int N, const int KL, const int KU, const float alpha, const float *A, const int lda, const float *X, const int incX, const float beta, float *Y, const int incY) {
//     size_t i;
//     #pragma omp parallel for
//     for (i = 0; i < n; i++) {
//         y[i * inc_y] = x[i * inc_x];
//     }
// }