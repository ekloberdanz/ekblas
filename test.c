#include <cblas.h>
#include <assert.h>
#include <math.h>

#include "ekblas.h"

int compare_floats(float a, float b) {
    const float epsilon = 0.001;
    return fabsf(a - b) <= epsilon * fabsf(a);
}

int compare_doubles(double a, double b) {
    const double epsilon = 0.0001;
    return fabs(a - b) <= epsilon * fabs(a);
}

int main () {
    float array_1[] = {1.0, 2.0, 3.0, 4.0};
    float array_2[] = {10.0, 20.0, 30.0, 40.0};
    float control;
    float result;

    control = cblas_sdot(4, array_1, 1, array_2, 1);
    printf("Result %f\n", control);

    result = ek_sdot(4, array_1, 1, array_2, 1);

    assert(compare_floats(control, result));

    puts("TESTS PASSED");
    return 0;
}