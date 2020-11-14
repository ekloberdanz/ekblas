CC=gcc
CFLAGS=-Wall -Wextra -std=gnu99 -fopenmp -lm
LDFLAGS=-lopenblas

.PHONY: all
all: test_single test_double

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