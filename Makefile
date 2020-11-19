CC=gcc
# Compile without openmp
CFLAGS=-Wall -Wextra -std=gnu99 -Ofast -march=native -mtune=native
# Compile with openmp
#CFLAGS=-Wall -Wextra -std=gnu99 -fopenmp -Ofast -march=native -mtune=native
# Compile with openmp gpu
#CFLAGS=-Wall -Wextra -std=gnu99 -fopenmp -foffload=nvptx-none -Ofast -march=native -mtune=native
LDFLAGS=-lopenblas -lm

.PHONY: all
all: test_single test_double benchmark_single benchmark_double

benchmark_single: benchmark_single.o ekblas.o
	$(CC) -o benchmark_single benchmark_single.o ekblas.o $(CFLAGS) $(LDFLAGS)

benchmark_single.o: benchmark_single.c ekblas.h
	$(CC) -o benchmark_single.o -c benchmark_single.c $(CFLAGS)

benchmark_double: benchmark_double.o ekblas.o
	$(CC) -o benchmark_double benchmark_double.o ekblas.o $(CFLAGS) $(LDFLAGS)

benchmark_double.o: benchmark_double.c ekblas.h
	$(CC) -o benchmark_double.o -c benchmark_double.c $(CFLAGS)

test_single: test_single.o ekblas.o
	$(CC) -o test_single test_single.o ekblas.o $(CFLAGS) $(LDFLAGS)

test_single.o: test_single.c ekblas.h
	$(CC) -o test_single.o -c test_single.c $(CFLAGS)

test_double: test_double.o ekblas.o
	$(CC) -o test_double test_double.o ekblas.o $(CFLAGS) $(LDFLAGS)

test_double.o: test_double.c ekblas.h
	$(CC) -o test_double.o -c test_double.c $(CFLAGS)

ekblas.o: ekblas.h ekblas.c
	$(CC) -o ekblas.o -c ekblas.c $(CFLAGS)

.PHONY: clean
clean:
	rm -f test_single
	rm -f test_double
	rm -f *.o