#include "debug.h"
#include "source_combine.h"

#include <stdio.h>
#include <inttypes.h>
#include <stdlib.h>

static int stdin_next(source_t *, uint64_t *res, uint64_t *p) {
  return scanf("%"SCNu64" %% %"SCNu64, res, p) == 2;
}

static void stdin_destroy(source_t *src) {
  free(src);
}

source_t *source_stdin_new(void) {
  source_t *src = malloc(sizeof *src);
  assert(src);
  *src = (source_t){ .next = stdin_next, .destroy = stdin_destroy, .state = NULL };
  return src;
}

