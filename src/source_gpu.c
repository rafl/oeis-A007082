#include "source_process.h"
#include "debug.h"

#ifdef USE_GPU

#include "gpu_compute.h"
#include "maths.h"
#include "mss.h"
#include "primes.h"

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define P_STRIDE (1ULL << 10)
#define GPU_BATCH_SIZE (1 << 16) // Process 64K vectors at a time

static uint64_t m_for(uint64_t n) { return 2 * ((n + 1) / 4) + 1; }

typedef struct {
  process_mode_t mode;
  uint64_t n, n_args, m;
  uint64_t *ps;
  size_t idx, np;
  bool quiet;
} gpu_state_t;

static int gpu_next(source_t *self, uint64_t *res, uint64_t *p_ret) {
  gpu_state_t *st = self->state;
  if (st->idx == st->np)
    return 0;

  uint64_t n = st->n, m = st->m, p = st->ps[st->idx];

  printf("DEBUG gpu_next: n=%lu, m=%lu, p=%lu\n", n, m, p);

  uint64_t w = mth_root_mod_p(p, m);
  printf("DEBUG: Found root w=%lu\n", w);

  if (!st->quiet)
    printf("GPU processing p=%" PRIu64 "\n", p);

  // Create GPU context
  bool jack_mode = st->mode == PROC_MODE_JACK_OFFSET;
  gpu_ctx_t *gpu_ctx = gpu_ctx_new(n, st->n_args, m, p, w, jack_mode);

  const size_t total_size = canon_iter_size(m, st->n_args);

  // Allocate batch buffer
  size_t *batch = malloc(GPU_BATCH_SIZE * m * sizeof(size_t));
  assert(batch);

  // Create iterator
  size_t *scratch = malloc((m + 1) * sizeof(size_t));
  assert(scratch);
  canon_iter_t it;
  canon_iter_new(&it, m, st->n_args, scratch);

  uint64_t acc = 0;
  size_t processed = 0;

  while (processed < total_size) {
    // Fill batch
    size_t batch_count = 0;
    while (batch_count < GPU_BATCH_SIZE && processed + batch_count < total_size) {
      if (!canon_iter_next(&it, &batch[batch_count * m]))
        break;
      batch_count++;
    }

    if (batch_count == 0)
      break;

    // Process batch on GPU
    uint64_t batch_result = gpu_process_batch(gpu_ctx, batch, batch_count);

    // Accumulate result
    acc += batch_result;
    if (acc >= p)
      acc -= p;

    processed += batch_count;

    if (!st->quiet && processed % 1000000 == 0) {
      printf("  Processed %zu / %zu (%.1f%%)\n", processed, total_size,
             100.0 * processed / total_size);
    }
  }

  free(scratch);
  free(batch);
  gpu_ctx_free(gpu_ctx);

  // Final normalization (divide by m^(n-1))
  uint64_t p_dash = (uint64_t)(-inv64_u64(p));
  uint64_t r = ((uint128_t)1 << 64) % p;
  uint64_t r2 = (uint128_t)r * r % p;
  uint64_t r3 = (uint128_t)r2 * r % p;

  uint64_t m_M = mont_mul(m, r2, p, p_dash);
  uint64_t denom =
      mont_inv(mont_pow(m_M, st->n_args - 1, r, p, p_dash), r3, p, p_dash);
  uint64_t result = mont_mul(mont_mul(acc, denom, p, p_dash), 1, p, p_dash);

  *p_ret = st->ps[st->idx++];
  *res = result;
  return 1;
}

static void gpu_destroy(source_t *self) {
  gpu_state_t *st = self->state;
  free(st->ps);
  free(st);
  free(self);
}

source_t *source_gpu_new(process_mode_t mode, uint64_t n, uint64_t m_id,
                         bool quiet) {
  if (!gpu_available()) {
    fprintf(stderr, "GPU not available\n");
    return NULL;
  }

  uint64_t m = m_for(n);
  assert(mode <= PROC_MODE_JACKEST);
  if (mode == PROC_MODE_JACKEST || mode == PROC_MODE_JACK_OFFSET) {
    assert("jack modes are only valid when n%4 == 3 && n > 3" &&
           (n > 3 && n % 4 == 3));
    m -= 2;
  }

  uint64_t n_args = (mode == PROC_MODE_JACK_OFFSET) ? n - 2 : n;
  size_t np;
  assert(m_id < P_STRIDE);
  uint64_t *ps = build_prime_list(n, m, m_id, P_STRIDE, &np);

  gpu_state_t *st = malloc(sizeof(*st));
  assert(st);
  *st = (gpu_state_t){.mode = mode,
                      .n = n,
                      .n_args = n_args,
                      .m = m,
                      .idx = 0,
                      .np = np,
                      .ps = ps,
                      .quiet = quiet};

  source_t *src = malloc(sizeof *src);
  assert(src);
  *src = (source_t){.next = gpu_next, .destroy = gpu_destroy, .state = st};
  return src;
}

#endif // USE_GPU
