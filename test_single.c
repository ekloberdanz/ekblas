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
    const float array_1[] = {1.0, 2.0, 3.0, -4.0};
    const float array_2[] = {10.0, -20.0, 30.0, 40.0};
    float control_array[] = {0.0, 0.0, 0.0, 0.0};
    float result_array[] = {0.0, 0.0, 0.0, 0.0};
    float tmp1[] = {0.0, 0.0, 0.0, 0.0};
    float tmp2[] = {0.0, 0.0, 0.0, 0.0};
    const float param[] = {1.0, 2.0, -3.0, -4.0, 0.3};
    float control;
    float result;

    control = cblas_sdsdot(4, 2.3, array_1, 1, array_2, 1);
    result = ek_sdsdot(4, 2.3, array_1, 1, array_2, 1);
    printf("control: %f, result: %f\n", control, result);
    assert(compare_floats(control, result));

    control = cblas_sdot(4, array_1, 1, array_2, 1);
    result = ek_sdot(4, array_1, 1, array_2, 1);
    printf("control: %f, result: %f\n", control, result);
    assert(compare_floats(control, result));


    control = cblas_sdot(4, array_1, 0, array_2, 1);
    result = ek_sdot(4, array_1, 0, array_2, 1);
    printf("control: %f, result: %f\n", control, result);
    assert(compare_floats(control, result));

    control = cblas_sasum(4, array_1, 1);
    result = ek_sasum(4, array_1, 1);
    printf("control: %f, result: %f\n", control, result);
    assert(compare_floats(control, result));

    memcpy(result_array, array_2, sizeof(array_2));
    memcpy(control_array, array_2, sizeof(array_2));
    cblas_saxpy(4, 2.3, array_1, 1, control_array, 1);
    ek_saxpy(4, 2.3, array_1, 1, result_array, 1);
    assert(compare_arrays_floats(control_array, result_array, 4));

    control = cblas_snrm2(4, array_1, 1);
    result = ek_snrm2(4, array_1, 1);
    printf("control: %f, result: %f\n", control, result);
    assert(compare_floats(control, result));

    memcpy(result_array, array_2, sizeof(array_2));
    memcpy(control_array, array_2, sizeof(array_2));
    cblas_sscal(4, 2.3, control_array, 1);
    ek_sscal(4, 2.3, result_array, 1);
    assert(compare_arrays_floats(control_array, result_array, 4));

    memcpy(result_array, array_1, sizeof(array_1));
    memcpy(control_array, array_2, sizeof(array_2));
    ek_sswap(4, control_array, 1, result_array, 1);
    assert(compare_arrays_floats(control_array, array_1, 4));
    assert(compare_arrays_floats(result_array, array_2, 4));

    memcpy(result_array, array_2, sizeof(array_2));
    memcpy(control_array, array_2, sizeof(array_2));
    cblas_scopy(4, array_1, 1, control_array, 1);
    ek_scopy(4, array_1, 1, result_array, 1);
    assert(compare_arrays_floats(control_array, result_array, 4));

    memcpy(result_array, array_1, sizeof(array_1));
    memcpy(control_array, array_2, sizeof(array_2));
    memcpy(tmp1, array_1, sizeof(array_1));
    memcpy(tmp2, array_2, sizeof(array_2));
    cblas_srot(4, result_array, 1, control_array, 1, 1.2, 1.3);
    ek_srot(4, tmp1, 1, tmp2, 1, 1.2, 1.3);
    assert(compare_arrays_floats(tmp1, result_array, 4));
    assert(compare_arrays_floats(tmp2, control_array, 4));

    float a1, a2, b1, b2, c1, c2, s1, s2;
    a1 = a2 = 15.7;
    b1 = b2 = 4.4;
    cblas_srotg(&a1, &b1, &c1, &s1); 
    ek_srotg(&a2, &b2, &c2, &s2);
    assert(compare_floats(a1, a2));
    assert(compare_floats(b1, b2));
    assert(compare_floats(c1, c2));
    assert(compare_floats(s1, s2));

    memcpy(result_array, array_1, sizeof(array_1));
    memcpy(control_array, array_2, sizeof(array_2));
    memcpy(tmp1, array_1, sizeof(array_1));
    memcpy(tmp2, array_2, sizeof(array_2));
    cblas_srotm(4, result_array, 1, control_array, 1, param);
    ek_srotm(4, tmp1, 1, tmp2, 1, param);
    assert(compare_arrays_floats(tmp1, result_array, 4));
    assert(compare_arrays_floats(tmp2, control_array, 4));

    float A[] = {
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0
    };

    float B[] = {
        7.0, 8.0,
        9.0, 10.0,
        11.0, 12.0
    };

    float C1[] = {
        0.0, 0.0,
        0.0, 0.0
    };

    float C2[] = {
        0.0, 0.0,
        0.0, 0.0
    };

    size_t m = 2;
    size_t n = 2;
    size_t k = 3;

    float alpha = 1.0;
    float beta = 0.0;

    ek_sgemm(m, n, k, alpha, A, B, beta, C1);

    puts("");
    print_matrix(C1, m, n);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, A, k, B, n, beta, C2, n);
    assert(compare_arrays_floats(C1, C2, 4));

    puts("TESTS PASSED");
    return 0;
}