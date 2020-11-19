#include <time.h>
#include <stdio.h>
#include <unistd.h>
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

int main(){

    // declare variables for timing routines
    clock_t t_start, t_end;
    size_t runtime;
    TIMEIT(
        sleep(2);,
        &runtime
    );
    printf("time result: %zu s\n", runtime);
    return 0;
}


