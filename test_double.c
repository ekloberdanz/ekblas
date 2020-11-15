#include <cblas.h>
#include <assert.h>
#include <math.h>
#include <string.h>

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
    // double precision
    const double array_1[] = {1.0, 2.0, 3.0, -4.0};
    const double array_2[] = {10.0, -20.0, 30.0, 40.0};
    const float array_1_single[] = {1.0, 2.0, 3.0, -4.0};
    const float array_2_single[] = {10.0, -20.0, 30.0, 40.0};
    double tmp1[] = {0.0, 0.0, 0.0, 0.0};
    double tmp2[] = {0.0, 0.0, 0.0, 0.0};
    const double param[] = {1.0, 2.0, -3.0, -4.0, 0.3};
    double control_array[] = {0.0, 0.0, 0.0, 0.0};
    double result_array[] = {0.0, 0.0, 0.0, 0.0};
    double control;
    double result;

    control = cblas_dsdot(4, array_1_single, 1, array_2_single, 1);
    result = ek_dsdot(4, array_1_single, 1, array_2_single, 1);
    printf("control: %f, result: %f\n", control, result);
    assert(compare_floats(control, result));

    control = cblas_ddot(4, array_1, 1, array_2, 1);
    result = ek_ddot(4, array_1, 1, array_2, 1);
    printf("control: %f, result: %f\n", control, result);
    assert(compare_doubles(control, result));


    control = cblas_ddot(4, array_1, 0, array_2, 1);
    result = ek_ddot(4, array_1, 0, array_2, 1);
    printf("control: %f, result: %f\n", control, result);
    assert(compare_doubles(control, result));

    control = cblas_dasum(4, array_1, 1);
    result = ek_dasum(4, array_1, 1);
    printf("control: %f, result: %f\n", control, result);
    assert(compare_doubles(control, result));

    memcpy(result_array, array_2, sizeof(array_2));
    memcpy(control_array, array_2, sizeof(array_2));
    cblas_daxpy(4, 2.3, array_1, 1, control_array, 1);
    ek_daxpy(4, 2.3, array_1, 1, result_array, 1);
    assert(compare_arrays_doubles(control_array, result_array, 4));

    control = cblas_dnrm2(4, array_1, 1);
    result = ek_dnrm2(4, array_1, 1);
    printf("control: %f, result: %f\n", control, result);
    assert(compare_doubles(control, result));

    memcpy(result_array, array_2, sizeof(array_2));
    memcpy(control_array, array_2, sizeof(array_2));
    cblas_dscal(4, 2.3, control_array, 1);
    ek_dscal(4, 2.3, result_array, 1);
    assert(compare_arrays_doubles(control_array, result_array, 4));

    memcpy(result_array, array_1, sizeof(array_1));
    memcpy(control_array, array_2, sizeof(array_2));
    ek_dswap(4, control_array, 1, result_array, 1);
    assert(compare_arrays_doubles(control_array, array_1, 4));
    assert(compare_arrays_doubles(result_array, array_2, 4));

    memcpy(result_array, array_2, sizeof(array_2));
    memcpy(control_array, array_2, sizeof(array_2));
    cblas_dcopy(4, array_1, 1, control_array, 1);
    ek_dcopy(4, array_1, 1, result_array, 1);
    assert(compare_arrays_doubles(control_array, result_array, 4));

    memcpy(result_array, array_1, sizeof(array_1));
    memcpy(control_array, array_2, sizeof(array_2));
    memcpy(tmp1, array_1, sizeof(array_1));
    memcpy(tmp2, array_2, sizeof(array_2));
    cblas_drot(4, result_array, 1, control_array, 1, 1.2, 1.3);
    ek_drot(4, tmp1, 1, tmp2, 1, 1.2, 1.3);
    assert(compare_arrays_doubles(tmp1, result_array, 4));
    assert(compare_arrays_doubles(tmp2, control_array, 4));

    memcpy(result_array, array_1, sizeof(array_1));
    memcpy(control_array, array_2, sizeof(array_2));
    memcpy(tmp1, array_1, sizeof(array_1));
    memcpy(tmp2, array_2, sizeof(array_2));
    cblas_drotm(4, result_array, 1, control_array, 1, param);
    ek_drotm(4, tmp1, 1, tmp2, 1, param);
    assert(compare_arrays_doubles(tmp1, result_array, 4));
    assert(compare_arrays_doubles(tmp2, control_array, 4));

    double A[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };

    double B[] = {
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    };

    double C1[] = {
        0.0, 0.0,
        0.0, 0.0
    };

    double C2[] = {
        0.0, 0.0,
        0.0, 0.0
    };

    size_t m = 2;
    size_t n = 2;
    size_t k = 3;

    double alpha = 1.0;
    double beta = 0.0;

    ek_dgemm(m, n, k, alpha, A, B, beta, C1);

    puts("");
    print_matrix(C1, m, n);
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C2, n);
    assert(compare_arrays_doubles(C1, C2, 4));

    puts("TESTS PASSED");
    return 0;
}