#ifndef EKBLAS_H
#define EKBLAS_H

#include <stddef.h>

float ek_sdsdot(const size_t n, const float alpha, const float *x, const size_t inc_x, const float *y, const size_t inc_y);
float ek_sdot(const size_t n, const float *x, const size_t inc_x, const float *y, const size_t inc_y);
float ek_sasum(const size_t n, const float *x, const size_t inc_x);
void ek_saxpy(const size_t n, const float alpha, const float *x, const size_t inc_x, float *y, const size_t inc_y);
float ek_snrm2(const size_t n, const float *x, const size_t inc_x);
void ek_sscal(const size_t n, const float alpha, float *x, const size_t inc_x);
void ek_sswap(const size_t n, float *x, const size_t inc_x, float *y, const size_t inc_y);
void ek_scopy(const size_t n, const float *x, const size_t inc_x, float *y, const size_t inc_y);
void ek_srot(const size_t n, float *x, const size_t inc_x, float *y, const size_t inc_y, const float c, const float s);
void ek_srotg(float *a, float *b, float *c, float *s);
void ek_srotm(const size_t n, float *x, const size_t inc_x, float *y, const size_t inc_y, const float *param);
void ek_sgemm(const size_t M, const size_t N, const size_t K, float alpha, float *A, float *B, float beta, float *C);

double ek_dsdot(const size_t n, const float *x, const size_t inc_x, const float *y, const size_t inc_y);
double ek_ddot(const size_t n, const double *x, const size_t inc_x, const double *y, const size_t inc_y);
double ek_dasum(const size_t n, const double *x, const size_t inc_x);
void ek_daxpy(const size_t n, const double alpha, const double *x, const size_t inc_x, double *y, const size_t inc_y);
double ek_dnrm2(const size_t n, const double *x, const size_t inc_x);
void ek_dscal(const size_t n, const double alpha, double *x, const size_t inc_x);
void ek_dswap(const size_t n, double *x, const size_t inc_x, double *y, const size_t inc_y);
void ek_dcopy(const size_t n, const double *x, const size_t inc_x, double *y, const size_t inc_y);
void ek_drot(const size_t n, double *x, const size_t inc_x, double *y, const size_t inc_y, const double c, const double s);
void ek_drotm(const size_t n, double *x, const size_t inc_x, double *y, const size_t inc_y, const double *param);
void ek_dgemm(const size_t M, const size_t N, const size_t K, double alpha, double *A, double *B, double beta, double *C);

#endif /* EKBLAS_H */