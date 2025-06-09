CFLAGS = -g -std=gnu18 -O3 -march=native -flto -fopenmp -Wall -Wextra $(shell pkg-config gmp --cflags)
LDFLAGS = $(shell pkg-config gmp --libs) -lm

oeis: oeis.c
	$(CC) $(CFLAGS) oeis.c $(LDFLAGS) -o oeis
