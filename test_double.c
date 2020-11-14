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

int main () {
    // double precision
    const double array_1[] = {1.0, 2.0, 3.0, -4.0};
    const double array_2[] = {10.0, -20.0, 30.0, 40.0};
    double control_array[] = {0.0, 0.0, 0.0, 0.0};
    double result_array[] = {0.0, 0.0, 0.0, 0.0};
    double control;
    double result;

    control = cblas_dsdot(4, array_1, 1, array_2, 1);
    result = ek_dsdot(4, array_1, 1, array_2, 1);
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

    puts("TESTS PASSED");
    return 0;
}