CFLAGS = -g -std=c18 -O3 -march=native -flto -Wall -Wextra $(shell pkg-config gmp --cflags)
LDFLAGS = $(shell pkg-config gmp --libs)

oeis: oeis.c
	cc $(CFLAGS) oeis.c $(LDFLAGS) -o oeis
