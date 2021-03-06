#include <cblas.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <sys/time.h>

#include "ekblas.h"

#define TIMEIT(stmt, time_buffer)                                             \
    do {                                                                      \
        struct timeval _ksm_start, _ksm_finish;                               \
        gettimeofday(&_ksm_start, NULL);                                      \
        {                                                                     \
            stmt;                                                             \
        }                                                                     \
        gettimeofday(&_ksm_finish, NULL);                                     \
        *time_buffer =                                                        \
            (1000000 * _ksm_finish.tv_sec + _ksm_finish.tv_usec) -            \
            (1000000 * _ksm_start.tv_sec + _ksm_start.tv_usec);               \
        *time_buffer /= 1000000;                                              \
    } while (0)


#define MICROTIMEIT(stmt, time_buffer)                                        \
    do {                                                                      \
        struct timeval _ksm_start, _ksm_finish;                               \
        gettimeofday(&_ksm_start, NULL);                                      \
        {                                                                     \
            stmt;                                                             \
        }                                                                     \
        gettimeofday(&_ksm_finish, NULL);                                     \
        *time_buffer =                                                        \
            (1000000 * _ksm_finish.tv_sec + _ksm_finish.tv_usec) -            \
            (1000000 * _ksm_start.tv_sec + _ksm_start.tv_usec);               \
    } while (0)

int compare_floats(float a, float b) {
    const float epsilon = 0.001;
    return fabsf(a - b) <= epsilon * fabsf(a);
}

int compare_doubles(double a, double b) {
    const double epsilon = 0.0001;
    return fabs(a - b) <= epsilon * fabs(a);
}

int compare_arrays_floats(const float a[], const float b [], size_t n) {
    size_t i;
    for (i = 0; i < n; i++) {
        if (compare_floats(a[i], b[i]) == 0) {
            return 0;
        }
    }
    return 1;
}

int compare_arrays_doubles(const double a[], const double b [], size_t n) {
    size_t i;
    for (i = 0; i < n; i++) {
        if (compare_floats(a[i], b[i]) == 0) {
            return 0;
        }
    }
    return 1;
}

void print_array_single(const float a[], size_t n) {
    size_t i;
    for (i = 0; i < n; i++) {
        printf("%f, ", a[i]);
    }
}

void print_matrix(const double *mat, size_t n, size_t m) {
    size_t i;

    for (i = 0; i < (n * m); i++) {
        if (((i + 1) % m) == 0) {
            printf("%f\n", mat[i]);
        } else {
            printf("%f ", mat[i]);
        }
    }
}

int main () {

    // set size of arrays
    const size_t M = 10000;
    const size_t N = 50000;
    const size_t K = 10000;
    const size_t SIZE = M * N;

    // allocate memory for large arrays for benchmarking code to cblas
    double *A = malloc(sizeof(double) * (M * K));
    double *B = malloc(sizeof(double) * (K * N));
    double *C = malloc(sizeof(double) * (M * N));
    float *B_single = malloc(sizeof(double) * (K * N));
    float *C_single = malloc(sizeof(double) * (M * N));

    // fill arrays with values
    size_t i;

    #pragma omp parallel for
    for (i = 0; i < (M * K); i++) {
        A[i] = rand();
    }

    #pragma omp parallel for
    for (i = 0; i < (K * N); i++) {
        B[i] = rand();
        B_single[i] = rand();
    }

    #pragma omp parallel for
    for (i = 0; i < (M * N); i++) {
        C[i] = rand();
        C_single[i] = rand();
    }

    const double param[] = {1.0, 2.0, -3.0, -4.0, 0.3};

    // declare variables for timing routines
    size_t runtime;

    // open file to write results to
    FILE *fp;
    fp = fopen("results_double.csv", "w");


     // dsdot
    MICROTIMEIT(
        cblas_dsdot(SIZE, C_single, 1, B_single, 1);,
        &runtime
    );
    printf("cblas_dsdot: %zu micro s\n", runtime);
    fprintf(fp, "%s", "dsdot");
    fprintf(fp, ",%zu", runtime);
    MICROTIMEIT(
        ek_dsdot(SIZE, C_single, 1, B_single, 1);,
        &runtime
    );
    printf("ek_dsdot: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);
    
    // ddot
    MICROTIMEIT(
        cblas_ddot(SIZE, C, 1, B, 1);,
        &runtime
    );
    printf("cblas_ddot: %zu micro s\n", runtime);
    fprintf(fp, "%s", "ddot");
    fprintf(fp, ",%zu", runtime);
    MICROTIMEIT(
        ek_ddot(SIZE, C, 1, B, 1);,
        &runtime
    );
    printf("ek_ddot: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // dasum
    MICROTIMEIT(
        cblas_dasum(SIZE, C, 1);,
        &runtime
    );
    printf("cblas_dasum: %zu micro s\n", runtime);
    fprintf(fp, "%s", "dasum");
    fprintf(fp, ",%zu", runtime);
    MICROTIMEIT(
        ek_dasum(SIZE, C, 1);,
        &runtime
    );
    printf("ek_dasum: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // daxpy
    MICROTIMEIT(
        cblas_daxpy(SIZE, 2.3, C, 1, B, 1);,
        &runtime
    );
    printf("cblas_daxpy: %zu micro s\n", runtime);
    fprintf(fp, "%s", "daxpy");
    fprintf(fp, ",%zu", runtime);
    MICROTIMEIT(
        ek_daxpy(SIZE, 2.3, C, 1, B, 1);,
        &runtime
    );
    printf("ek_daxpy: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // dnrm2
    MICROTIMEIT(
        cblas_dnrm2(SIZE, C, 1);,
        &runtime
    );
    printf("cblas_dnrm2: %zu micro s\n", runtime);
    fprintf(fp, "%s", "dnrm2");
    fprintf(fp, ",%zu", runtime);
    MICROTIMEIT(
        ek_dnrm2(SIZE, C, 1);,
        &runtime
    );
    printf("ek_dnrm2: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // dscal
    MICROTIMEIT(
        cblas_dscal(SIZE, 2.3, B, 1);,
        &runtime
    );
    printf("cblas_dscal: %zu micro s\n", runtime);
    fprintf(fp, "%s", "dscal");
    fprintf(fp, ",%zu", runtime);
    MICROTIMEIT(
        ek_dscal(SIZE, 2.3, B, 1);,
        &runtime
    );
    printf("ek_dscal: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // dswap
    MICROTIMEIT(
        cblas_dswap(SIZE, C, 1, B, 1);,
        &runtime
    );
    printf("cblas_dswap: %zu micro s\n", runtime);
    fprintf(fp, "%s", "dswap");
    fprintf(fp, ",%zu", runtime);
    MICROTIMEIT(
        ek_dswap(SIZE, C, 1, B, 1);,
        &runtime
    );
    printf("ek_dswap: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // dcopy
    MICROTIMEIT(
        cblas_dcopy(SIZE, C, 1, B, 1);,
        &runtime
    );
    printf("cblas_dcopy: %zu micro s\n", runtime);
    fprintf(fp, "%s", "dcopy");
    fprintf(fp, ",%zu", runtime);
    MICROTIMEIT(
        ek_dcopy(SIZE, C, 1, B, 1);,
        &runtime
    );
    printf("ek_dcopy: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // drot
    MICROTIMEIT(
        cblas_drot(SIZE, C, 1, B, 1, 1.2, 1.3);,
        &runtime
    );
    printf("cblas_drot: %zu micro s\n", runtime);
    fprintf(fp, "%s", "drot");
    fprintf(fp, ",%zu", runtime);
    MICROTIMEIT(
        ek_drot(SIZE, C, 1, B, 1, 1.2, 1.3);,
        &runtime
    );
    printf("ek_drot: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // drotm
    MICROTIMEIT(
        cblas_drotm(SIZE, C, 1, B, 1, param);,
        &runtime
    );
    printf("cblas_drotm: %zu micro s\n", runtime);
    fprintf(fp, "%s", "drotm");
    fprintf(fp, ",%zu", runtime);
    MICROTIMEIT(
        ek_drotm(SIZE, C, 1, B, 1, param);,
        &runtime
    );
    printf("ek_drotm: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // dgemm
    size_t m = M; // rows in A and C
    size_t n = N; // columns in B and C
    size_t k = K; // rows in B and columns in A

    double alpha = 1.0;
    double beta = 0.0;

    TIMEIT(
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);,
        &runtime
    );
    printf("cblas_dgemm: %zu s\n", runtime);
    fprintf(fp, "%s", "dgemm");
    fprintf(fp, ",%zu", runtime);
    TIMEIT(
        ek_dgemm(m, n, k, alpha, A, B, beta, C);,
        &runtime
    );
    printf("ek_dgemm: %zu s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // close results file
    fclose(fp);

    puts("BENCHMARKS COMPLETE");

    // free memory of the arrays
    free(C);
    free(B);
    free(A);
    free(B_single);
    free(C_single);


    return 0;
}