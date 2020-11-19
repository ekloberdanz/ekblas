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


#define MICROTIMEIT(stmt, time_buffer)                                             \
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
        printf("%ld, ", a[i]);
    }
}

void print_matrix(const float *mat, size_t n, size_t m) {
    size_t i;

    for (i = 0; i < (n * m); i++) {
        if (((i + 1) % m) == 0) {
            printf("%ld\n", mat[i]);
        } else {
            printf("%ld ", mat[i]);
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
    float *A = malloc(sizeof(float) * (M * K));
    float *B = malloc(sizeof(float) * (K * N));
    float *C = malloc(sizeof(float) * (M * N));

    // fill arrays with values
    size_t i;

    #pragma omp parallel for
    for (i = 0; i < (M * K); i++) {
        A[i] = rand();
    }

    #pragma omp parallel for
    for (i = 0; i < (K * N); i++) {
        B[i] = rand();
    }

    #pragma omp parallel for
    for (i = 0; i < (M * N); i++) {
        C[i] = rand();
    }

    const float param[] = {1.0, 2.0, -3.0, -4.0, 0.3};

    // declare variables for timing routines
    size_t runtime;

    // open file to write results to
    FILE *fp;
    fp = fopen("results_single.csv", "w");

    // benchmark each routine implementation to cblas

    // sdsdot
   
    fprintf(fp, "%s", "sdsdot");
    MICROTIMEIT(
        ek_sdsdot(SIZE, 2.3, C, 1, B, 1);,
        &runtime
    );
    printf("ek_sdsdot: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);
    
    // sdot
    
    fprintf(fp, "%s", "sdot");
    MICROTIMEIT(
        ek_sdot(SIZE, C, 1, B, 1);,
        &runtime
    );
    printf("ek_sdot: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // sasum
   
    fprintf(fp, "%s", "sasum");
    MICROTIMEIT(
        ek_sasum(SIZE, C, 1);,
        &runtime
    );
    printf("ek_sasum: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // saxpy
    fprintf(fp, "%s", "saxpy");
    MICROTIMEIT(
        ek_saxpy(SIZE, 2.3, C, 1, B, 1);,
        &runtime
    );
    printf("ek_saxpy: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // snrm2
    fprintf(fp, "%s", "snrm2");
    MICROTIMEIT(
        ek_snrm2(SIZE, C, 1);,
        &runtime
    );
    printf("ek_snrm2: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // sscal
    fprintf(fp, "%s", "sscal");
    MICROTIMEIT(
        ek_sscal(SIZE, 2.3, B, 1);,
        &runtime
    );
    printf("ek_sscal: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // sswap
    fprintf(fp, "%s", "sswap");
    MICROTIMEIT(
        ek_sswap(SIZE, C, 1, B, 1);,
        &runtime
    );
    printf("ek_sswap: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // scopy
    fprintf(fp, "%s", "scopy");
    MICROTIMEIT(
        ek_scopy(SIZE, C, 1, B, 1);,
        &runtime
    );
    printf("ek_scopy: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // srot
    fprintf(fp, "%s", "srot");
    MICROTIMEIT(
        ek_srot(SIZE, C, 1, B, 1, 1.2, 1.3);,
        &runtime
    );
    printf("ek_srot: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);


    // srotg
    float a1, a2, b1, b2, c2, s2;
    a1 = a2 = 15.7;
    b1 = b2 = 4.4;

    fprintf(fp, "%s", "srotg");
    MICROTIMEIT(
        ek_srotg(&a2, &b2, &c2, &s2);,
        &runtime
    );
    printf("ek_srotg: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // srotm
    fprintf(fp, "%s", "srotm");
    MICROTIMEIT(
        ek_srotm(SIZE, C, 1, B, 1, param);,
        &runtime
    );
    printf("ek_srotm: %zu micro s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // sgemm
    size_t m = M; // rows in A and C
    size_t n = N; // columns in B and C
    size_t k = K; // rows in B and columns in A

    float alpha = 1.0;
    float beta = 0.0;

    fprintf(fp, "%s", "sgemm");
    TIMEIT(
        ek_sgemm(m, n, k, alpha, A, B, beta, C);,
        &runtime
    );
    printf("ek_sgemm: %zu s\n", runtime);
    fprintf(fp, ",%ld\n", runtime);

    // close results file
    fclose(fp);

    puts("BENCHMARKS COMPLETE");

    // free memory of the arrays
    free(A);
    free(B);
    free(C);

    return 0;
}