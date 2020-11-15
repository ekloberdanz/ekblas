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

void print_matrix(const float *mat, size_t n, size_t m) {
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

    // single precision
    const size_t SIZE = 100000;
    const size_t M = 1000;
    const size_t N = 5000;
    const size_t K = 1000;

    // allocate memory for large arrays for benchmarking code to cblas
    float *array_1 = malloc(sizeof(float) * SIZE);
    float *array_2 = malloc(sizeof(float) * SIZE);
    float *control_array = malloc(sizeof(float) * SIZE);
    float *result_array = malloc(sizeof(float) * SIZE);
    float *tmp1 = malloc(sizeof(float) * SIZE);
    float *tmp2 = malloc(sizeof(float) * SIZE);
    const float param[] = {1.0, 2.0, -3.0, -4.0, 0.3};
    float *control = malloc(sizeof(float) * SIZE);
    float *result = malloc(sizeof(float) * SIZE);
    float *A = malloc(sizeof(float) * (M * K));
    float *B = malloc(sizeof(float) * (K * N));
    float *C1 = malloc(sizeof(float) * (M * N));
    float *C2 = malloc(sizeof(float) * (M * N));


    // fill arrays with values
    size_t i;
    #pragma omp parallel for
    for (i = 0; i < SIZE; i++) {
        array_1[i] = rand();
        array_2[i] = rand();
        control_array[i] = 0;
        result_array[i] = 0;
        tmp1[i] = 0;
        tmp2[i] = 0;
    }

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
        C1[i] = 0;
        C2[i] = 0;
    }

    // declare variables for timing routines
    clock_t t_start, t_end;
    double t_delta;

    // open file to write results to
    FILE *fp;
    fp = fopen("results.csv", "w");

    // benchmark each routine implementation to cblas
    t_start = clock();
    *control = cblas_sdsdot(SIZE, 2.3, array_1, 1, array_2, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "sdsdot");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    *result = ek_sdsdot(SIZE, 2.3, array_1, 1, array_2, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);
    
    t_start = clock();
    *control = cblas_sdot(SIZE, array_1, 1, array_2, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "sdot");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    *result = ek_sdot(SIZE, array_1, 1, array_2, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);

    t_start = clock();
    *control = cblas_sasum(SIZE, array_1, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "sasum");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    *result = ek_sasum(SIZE, array_1, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);

    memcpy(result_array, array_2, sizeof(*array_2));
    memcpy(control_array, array_2, sizeof(*array_2));
    t_start = clock();
    cblas_saxpy(SIZE, 2.3, array_1, 1, control_array, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "saxpy");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_saxpy(SIZE, 2.3, array_1, 1, result_array, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);

    t_start = clock();
    *control = cblas_snrm2(SIZE, array_1, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "snrm2");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    *result = ek_snrm2(SIZE, array_1, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);

    memcpy(result_array, array_2, sizeof(*array_2));
    memcpy(control_array, array_2, sizeof(*array_2));
    t_start = clock();
    cblas_sscal(SIZE, 2.3, control_array, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "sscal");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_sscal(SIZE, 2.3, result_array, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);


    memcpy(result_array, array_1, sizeof(*array_1));
    memcpy(control_array, array_2, sizeof(*array_2));
    t_start = clock();
    cblas_sswap(SIZE, control_array, 1, result_array, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "sswap");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_sswap(SIZE, control_array, 1, result_array, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);


    memcpy(result_array, array_2, sizeof(*array_2));
    memcpy(control_array, array_2, sizeof(*array_2));
    t_start = clock();
    cblas_scopy(SIZE, array_1, 1, control_array, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "scopy");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_scopy(SIZE, array_1, 1, result_array, 1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);

    memcpy(result_array, array_1, sizeof(*array_1));
    memcpy(control_array, array_2, sizeof(*array_2));
    memcpy(tmp1, array_1, sizeof(*array_1));
    memcpy(tmp2, array_2, sizeof(*array_2));
    t_start = clock();
    cblas_srot(SIZE, result_array, 1, control_array, 1, 1.2, 1.3);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "srot");
    fprintf(fp, ",%f", t_delta);
    t_start = clock();
    ek_srot(SIZE, tmp1, 1, tmp2, 1, 1.2, 1.3);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);


    float a1, a2, b1, b2, c1, c2, s1, s2;
    a1 = a2 = 15.7;
    b1 = b2 = 4.4;

    t_start = clock();
    cblas_srotg(&a1, &b1, &c1, &s1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "srotg");
    fprintf(fp, ",%f", t_delta);
    t_start = clock(); 
    ek_srotg(&a2, &b2, &c2, &s2);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);


    memcpy(result_array, array_1, sizeof(*array_1));
    memcpy(control_array, array_2, sizeof(*array_2));
    memcpy(tmp1, array_1, sizeof(*array_1));
    memcpy(tmp2, array_2, sizeof(*array_2));
    t_start = clock();
    cblas_srotm(SIZE, result_array, 1, control_array, 1, param);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "srotm");
    fprintf(fp, ",%f", t_delta);
    t_start = clock(); 
    ek_srotm(SIZE, tmp1, 1, tmp2, 1, param);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);


    size_t m = M; // rows in A and C
    size_t n = N; // columns in B and C
    size_t k = K; // rows in B and columns in A

    float alpha = 1.0;
    float beta = 0.0;

    t_start = clock();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C2, n);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time control: %f\n", t_delta);
    fprintf(fp, "%s", "sgemm");
    fprintf(fp, ",%f", t_delta);
    t_start = clock(); 
    ek_sgemm(m, n, k, alpha, A, B, beta, C1);
    t_end = clock() - t_start;
    t_delta = ((double)t_end)/CLOCKS_PER_SEC;
    printf("time result: %f\n", t_delta);
    fprintf(fp, ",%f\n", t_delta);

    // close results file
    fclose(fp);

    puts("BENCHMARKS COMPLETE");

    // free memory of the arrays
    free(array_1);
    free(array_2);
    free(control_array);
    free(result_array);
    free(tmp1);
    free(tmp2);
    free(result);
    free(control);
    free(A);
    free(B);
    free(C1);
    free(C2);

    return 0;
}