#include "mss.h"
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t m_for(uint64_t n) { return 2 * ((n + 1) / 4) + 1; }

static void usage(const char *prog) {
  fprintf(stderr, "Usage: %s [-q] <n> <depth>\n", prog);
  fprintf(stderr, "Generate prefix vectors at specified depth for n\n");
  fprintf(stderr, "  -q          Quiet mode: don't print prefixes/necklaces\n");
  fprintf(stderr, "  depth must be in [1, m] where m = 2*((n+1)/4)+1\n");
  fprintf(stderr, "  depth == m generates complete necklaces\n");
  fprintf(stderr, "  depth < m generates prefixes of that length\n");
  exit(1);
}

int main(int argc, char **argv) {
  bool quiet = false;
  int arg_offset = 1;

  if (argc > 1 && strcmp(argv[1], "-q") == 0) {
    quiet = true;
    arg_offset = 2;
  }

  if (argc != arg_offset + 2) {
    usage(argv[0]);
  }

  char *endptr;
  uint64_t n = strtoull(argv[arg_offset], &endptr, 10);
  if (*endptr != '\0' || n == 0) {
    fprintf(stderr, "Invalid n: %s\n", argv[arg_offset]);
    usage(argv[0]);
  }

  uint64_t depth = strtoull(argv[arg_offset + 1], &endptr, 10);
  if (*endptr != '\0' || depth == 0) {
    fprintf(stderr, "Invalid depth: %s\n", argv[arg_offset + 1]);
    usage(argv[0]);
  }

  uint64_t m = m_for(n);

  if (depth > m) {
    fprintf(stderr, "Error: depth %lu exceeds m=%lu\n", depth, m);
    exit(1);
  }

  if (!quiet) {
    printf("n=%lu, m=%lu, depth=%lu\n\n", n, m, depth);
  }

  size_t scratch[m + 1], vec[m];

  canon_iter_t iter;
  canon_iter_new(&iter, m, n, scratch, depth);

  uint64_t prefix_count = 0;
  uint64_t productive_count = 0;
  uint64_t necklace_count = 0;

  while (canon_iter_next(&iter, vec)) {
    prefix_count++;
    uint64_t prefix_necklaces = 0;

    if (!quiet) {
      printf("[");
      for (size_t i = 0; i < depth; i++) {
        printf("%zu", vec[i]);
        if (i < depth - 1)
          printf(", ");
      }
      printf("]\n");
    }

    if (depth < m) {
      size_t sub_scratch[m + 1], sub_vec[m];
      canon_iter_t sub_iter;

      canon_iter_from_prefix(&sub_iter, m, n, sub_scratch, vec, depth);

      while (canon_iter_next(&sub_iter, sub_vec)) {
        prefix_necklaces++;
        if (!quiet) {
          printf("  [");
          for (size_t i = 0; i < m; i++) {
            printf("%zu", sub_vec[i]);
            if (i < m - 1)
              printf(", ");
          }
          printf("]\n");
        }
      }
    } else {
      prefix_necklaces = 1;
    }

    necklace_count += prefix_necklaces;
    if (prefix_necklaces > 0) {
      productive_count++;
    }
  }

  printf("\nTotal prefixes: %lu\n", prefix_count);
  printf("Productive prefixes: %lu (%.1f%%)\n", productive_count,
         prefix_count > 0 ? 100.0 * productive_count / prefix_count : 0.0);
  printf("Empty prefixes: %lu\n", prefix_count - productive_count);
  printf("Total necklaces: %lu\n", necklace_count);

  return 0;
}
