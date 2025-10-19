#include "bracelet.h"
#include "debug.h"
#include "mss.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t m_for(uint64_t n) { return 2 * ((n + 1) / 4) + 1; }

static uint64_t parse_uint(const char *s) {
  char *e;
  uint64_t n = strtoull(s, &e, 10);
  if (s == e || *e != 0) {
    fprintf(stderr, "invalid uint: %s\n", s);
    abort();
  }
  return n;
}

static void print_vec(const size_t *vec, size_t m) {
  printf("[");
  for (size_t i = 0; i < m; ++i) {
    printf("%zu", vec[i]);
    if (i < m - 1)
      printf(", ");
  }
  printf("]");
}

static void print_reflection(const size_t *vec, size_t m) {
  printf("[");
  for (size_t i = 0; i < m; ++i) {
    size_t val = (i == 0) ? vec[0] : vec[m - i];
    printf("%zu", val);
    if (i < m - 1)
      printf(", ");
  }
  printf("]");
}

int main(int argc, char **argv) {
  bool use_bracelets = true;
  bool benchmark = false;
  int arg_idx = 1;

  if (argc < 2 || argc > 4)
    abort();

  while (arg_idx < argc && argv[arg_idx][0] == '-') {
    if (strcmp(argv[arg_idx], "-b") == 0) {
      use_bracelets = true;
      arg_idx++;
    } else if (strcmp(argv[arg_idx], "-n") == 0) {
      use_bracelets = false;
      arg_idx++;
    } else if (strcmp(argv[arg_idx], "--bench") == 0) {
      benchmark = true;
      arg_idx++;
    } else {
      abort();
    }
  }

  if (arg_idx >= argc)
    abort();

  uint64_t n = parse_uint(argv[arg_idx]);
  uint64_t m = m_for(n);
  size_t scratch[m + 1], vec[m + 1];
  uint64_t count = 0;

  if (use_bracelets) {
    bracelet_iter_t it;
    bracelet_iter_new(&it, m, n, scratch);

    while (bracelet_iter_next(&it, vec)) {
      if (!benchmark) {
        bool is_chiral = vec[0] != 0;
        printf("%s: ", is_chiral ? "chiral  " : "achiral ");
        print_vec(vec + 1, m);
        if (is_chiral) {
          printf(" and ");
          print_reflection(vec + 1, m);
        }
        printf("\n");
      }
      ++count;
    }

    if (!benchmark)
      printf("\nTotal bracelets: %" PRIu64 "\n", count);
    else
      printf("%" PRIu64 "\n", count);
  } else {
    canon_iter_t it;
    canon_iter_new(&it, m, n, scratch);

    while (canon_iter_next(&it, vec)) {
      if (!benchmark) {
        print_vec(vec, m);
        printf("\n");
      }
      ++count;
    }

    if (!benchmark)
      printf("\nTotal necklaces: %" PRIu64 "\n", count);
    else
      printf("%" PRIu64 "\n", count);
  }
  return 0;
}
