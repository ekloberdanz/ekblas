#ifndef EKBLAS_H
#define EKBLAS_H

#include <stddef.h>

float ek_sdot(const size_t n, const float *x, const size_t inc_x, const float *y, const size_t inc_y);
float ek_sasum(const size_t n, const float *x, const size_t inc_x);
void ek_saxpy(const size_t n, const float alpha, const float *x, const size_t inc_x, float *y, const size_t inc_y);

#endif /* EKBLAS_H */