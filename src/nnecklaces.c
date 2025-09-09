#include "debug.h"
#include "mss.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

static uint64_t parse_uint(const char *s) {
  char *e;
  uint64_t n = strtoull(s, &e, 10);
  if (s == e || *e != 0) {
    fprintf(stderr, "invalid uint: %s\n", s);
    abort();
  }
  return n;
}

int main (int argc, char **argv) {
  assert(argc == 3);
  printf("%"PRIu64"\n", canon_iter_size(parse_uint(argv[1]), parse_uint(argv[2])));
}
