#include "source_combine.h"
#include "debug.h"

#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

static int stdin_next(source_t *, uint64_t *res, uint64_t *p) {
  while (1) {
    char *l;
    size_t cap = 0;
    ssize_t n = getline(&l, &cap, stdin);
    if (n < 0) {
      free(l);
      return n;
    }
    char *s = l;
    while (*s && isspace((unsigned char)*s))
      s++;
    if (*s == '\0' || *s == '\n' || *s == '#') {
      free(l);
      continue;
    }

    bool ok = sscanf(s, "%" SCNu64 " %% %" SCNu64, res, p) == 2;
    free(l);
    return ok;
  }
}

static void stdin_destroy(source_t *src) { free(src); }

source_t *source_stdin_new(void) {
  source_t *src = malloc(sizeof *src);
  assert(src);
  *src =
      (source_t){.next = stdin_next, .destroy = stdin_destroy, .state = NULL};
  return src;
}
