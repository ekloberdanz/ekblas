CC=gcc
CFLAGS=-Wall -Wextra -std=gnu99 -fopenmp -lm
LDFLAGS=-lopenblas

test: test.o ekblas.o
	$(CC) -o test test.o ekblas.o $(CFLAGS) $(LDFLAGS)

test.o: test.c ekblas.h
	$(CC) -o test.o -c test.c $(CFLAGS)

ekblas.o: ekblas.h ekblas.c
	$(CC) -o ekblas.o -c ekblas.c $(CFLAGS)

.PHONY: clean
clean:
	rm -f test
	rm -f *.o