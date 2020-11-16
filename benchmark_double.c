#include <cblas.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

#include "ekblas.h"

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
    clock_t t_start, t_end;
    double t_delta;

    // open file to write results to
    FILE *fp;
    fp = fopen("results_double.csv", "w");

    // benchmark each routine implementation to cblas
    t_start = clock();
    cblas_dsdot(SIZE, C_single, 1, B_single, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "dsdot");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_dsdot(SIZE, C_single, 1, B_single, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);
    
    t_start = clock();
    cblas_ddot(SIZE, C, 1, B, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "ddot");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_ddot(SIZE, C, 1, B, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);

    t_start = clock();
    cblas_dasum(SIZE, C, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "dasum");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_dasum(SIZE, C, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);


    t_start = clock();
    cblas_daxpy(SIZE, 2.3, C, 1, B, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "daxpy");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_daxpy(SIZE, 2.3, C, 1, B, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);

    t_start = clock();
    cblas_dnrm2(SIZE, C, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "dnrm2");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_dnrm2(SIZE, C, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);

    t_start = clock();
    cblas_dscal(SIZE, 2.3, B, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "dscal");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_dscal(SIZE, 2.3, B, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);

    t_start = clock();
    cblas_dswap(SIZE, C, 1, B, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "dswap");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_dswap(SIZE, C, 1, B, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);


    t_start = clock();
    cblas_dcopy(SIZE, C, 1, B, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "dcopy");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_dcopy(SIZE, C, 1, B, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);


    t_start = clock();
    cblas_drot(SIZE, C, 1, B, 1, 1.2, 1.3);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "drot");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_drot(SIZE, C, 1, B, 1, 1.2, 1.3);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);

    t_start = clock();
    cblas_drotm(SIZE, C, 1, B, 1, param);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "drotm");
    fprintf(fp, ",%f", t_delta);
    t_start = clock(); 
    ek_drotm(SIZE, C, 1, B, 1, param);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);


    size_t m = M; // rows in A and C
    size_t n = N; // columns in B and C
    size_t k = K; // rows in B and columns in A

    double alpha = 1.0;
    double beta = 0.0;

    t_start = clock();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C, n);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "dgemm");
    fprintf(fp, ",%f", t_delta);
    t_start = clock(); 
    ek_dgemm(m, n, k, alpha, A, B, beta, C);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);

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