#include "oeis.h"
#include "debug.h"
#include "maths.h"
#include "primes.h"
#include "mss.h"
#include "combine.h"
#include "source_combine.h"
#include "source_process.h"

#include <stdio.h>
#include <inttypes.h>
#include <gmp.h>
#include <stdlib.h>
#include <unistd.h>

typedef enum {
  MODE_NONE = 0,
  MODE_PROCESS = (1 << 0),
  MODE_COMBINE = (1 << 1),
  MODE_BOTH = MODE_PROCESS|MODE_COMBINE,
  MODE_LAST = MODE_BOTH+1,
} prog_mode_t;

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
  uint64_t n = 13, m_id = 0;
  prog_mode_t mode = MODE_NONE;

  for (;;) {
    int c = getopt(argc, argv, "m:pc");
    if (c == -1) break;

    switch (c) {
      case 'm': m_id = parse_uint(optarg); break;
      case 'p': mode |= MODE_PROCESS; break;
      case 'c': mode |= MODE_COMBINE; break;
    }
  }
  assert(mode < MODE_LAST);
  if (mode == MODE_NONE) mode = MODE_BOTH;

  if (argc > optind)
    n = parse_uint(argv[optind]);

  source_t *src = (mode & MODE_PROCESS) ? source_process_new(n, m_id) : source_stdin_new();
  comb_ctx_t *crt = (mode & MODE_COMBINE) ? comb_ctx_new() : NULL;

  bool converged = false;
  size_t i = 0;
  uint64_t res, p;
  while (src->next(src, &res, &p) > 0) {
    if (mode & MODE_COMBINE) {
      converged = comb_ctx_add(crt, res, p);

      if (i > 0) {
        gmp_printf("e(%"PRIu64") %s %Zd\n  after %zu primes, mod %Zd\n",
                   n, converged ? "=" : ">=", crt->X, i+1, crt->M);
        if (converged && mode & MODE_PROCESS) break;
      }
      ++i;
    }
  }
  src->destroy(src);

  if (mode & MODE_COMBINE) {
    if (!converged)
      gmp_printf("(INCOMPLETE) e_n = %Zd (mod %Zd)\n", crt->X, crt->M);
    comb_ctx_free(crt);
  }

  return 0;
}
