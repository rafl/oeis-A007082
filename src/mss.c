#include "mss.h"
#include "debug.h"
#include "maths.h"

#include <stdlib.h>
#include <string.h>

void canon_iter_new(canon_iter_t *it, size_t m, size_t tot, size_t *scratch) {
  it->m = m;
  it->tot = tot;
  it->scratch = scratch;
  memset(it->scratch, 0, (m + 1) * sizeof(size_t));

  it->t = 1; // position
  it->p = 1; // period length
  it->sum = 0;
  it->stage = ITER_STAGE_DESCEND;
  it->min_depth = 0;  // Start from root
}

// Bounded DFS to build frontier at specified depth
void canon_iter_frontier(canon_iter_t *it, size_t depth, void *save_buf,
                        size_t save_buf_len, frontier_emit_cb_t emit_cb,
                        void *user) {
  const size_t m = it->m;
  const size_t tot = it->tot;
  size_t *a = it->scratch;

  for (;;) {
    switch (it->stage) {
    case ITER_STAGE_DESCEND: {
      // If we've reached the depth boundary, emit and backtrack
      if (it->t > depth) {
        // Set min_depth so resumed iterator stops at this depth
        size_t saved_min = it->min_depth;
        it->min_depth = depth;
        size_t len = canon_iter_save(it, save_buf, save_buf_len);
        it->min_depth = saved_min;  // Restore for frontier traversal

        emit_cb(save_buf, len, user);
        it->stage = ITER_STAGE_BACKTRACK;
        break;
      }

      if (it->t > m) { // leaf - just backtrack
        it->stage = ITER_STAGE_BACKTRACK;
        break;
      }

      size_t v = a[it->t - it->p];
      if (it->sum + v <= tot) {
        a[it->t] = v;
        it->sum += v;
        ++it->t;
        break;
      }

      it->stage = ITER_STAGE_LOOP;
      a[it->t] = v + 1;
      break;
    }

    case ITER_STAGE_LOOP: {
      size_t v = a[it->t];
      if (v > tot - it->sum) {
        it->stage = ITER_STAGE_BACKTRACK;
        break;
      }

      it->sum += v;
      ++it->t;
      it->p = it->t - 1;
      it->stage = ITER_STAGE_DESCEND;
      break;
    }

    case ITER_STAGE_BACKTRACK:
      --it->t;
      if (it->t == 0)
        return; // Done with frontier

      it->sum -= a[it->t];
      a[it->t] += 1;
      it->stage = ITER_STAGE_LOOP;
      break;
    }
  }
}

// t p sum stage min_depth scratch...(m+1)
size_t canon_iter_save(canon_iter_t *it, void *buf, size_t len) {
  size_t n = (5 + it->m + 1) * sizeof(uint64_t);
  assert(len >= n);
  uint64_t *out = buf;
  memcpy(out, (uint64_t[]){it->t, it->p, it->sum, it->stage, it->min_depth},
         5 * sizeof(uint64_t));
  memcpy(out + 5, it->scratch, (it->m + 1) * sizeof(uint64_t));
  return n;
}

void canon_iter_resume(canon_iter_t *it, size_t m, size_t tot, size_t *scratch,
                       const void *buf, size_t len) {
  assert(len >= (5 + m + 1) * sizeof(uint64_t));
  const uint64_t *in = buf;
  it->m = m;
  it->tot = tot;
  it->t = in[0];
  it->p = in[1];
  it->sum = in[2];
  it->stage = in[3];
  it->min_depth = in[4];
  it->scratch = scratch;
  memcpy(it->scratch, in + 5, (m + 1) * sizeof(uint64_t));
}

bool canon_iter_next(canon_iter_t *it, size_t *vec) {
  const size_t m = it->m;
  const size_t tot = it->tot;
  size_t *a = it->scratch;

  for (;;) {
    switch (it->stage) {
    case ITER_STAGE_DESCEND: {
      if (it->t > m) { // leaf
        it->stage = ITER_STAGE_BACKTRACK;
        if (m % it->p == 0 && it->sum == tot) {
          vec[0] = a[m];
          memcpy(vec + 1, a + 1, (m - 1) * sizeof(size_t));
          return true;
        }
        break;
      }

      size_t v = a[it->t - it->p];
      if (it->sum + v <= tot) {
        a[it->t] = v;
        it->sum += v;
        ++it->t;
        break;
      }

      it->stage = ITER_STAGE_LOOP;
      a[it->t] = v + 1;
      break;
    }

    case ITER_STAGE_LOOP: {
      size_t v = a[it->t];
      if (v > tot - it->sum) {
        it->stage = ITER_STAGE_BACKTRACK;
        break;
      }

      it->sum += v;
      ++it->t;
      it->p = it->t - 1;
      it->stage = ITER_STAGE_DESCEND;
      break;
    }

    case ITER_STAGE_BACKTRACK:
      --it->t;
      if (it->t <= it->min_depth)
        return false;  // Stop at minimum depth

      it->sum -= a[it->t];
      a[it->t] += 1;
      it->stage = ITER_STAGE_LOOP;
      break;
    }
  }
}

static size_t gcd(size_t a, size_t b) {
  while (b) {
    size_t t = a % b;
    a = b;
    b = t;
  }
  return a;
}

static size_t phi(size_t d) {
  size_t r = d;
  for (size_t p = 2; p * p <= d; ++p) {
    if (d % p == 0) {
      while (d % p == 0)
        d /= p;
      r -= r / p;
    }
  }
  if (d > 1)
    r -= r / d;
  return r;
}

static uint64_t binom(size_t n, size_t k) {
  if (k > n)
    return 0;
  if (k > n - k)
    k = n - k;
  uint128_t num = 1, den = 1;
  for (size_t i = 1; i <= k; ++i) {
    num *= n - k + i;
    den *= i;
  }
  return (uint64_t)(num / den);
}

// number of different elements in cannonical form
//(i.e. considering just the set of args, not the order)
size_t canon_iter_size(size_t m, size_t n) {
  size_t g = gcd(m, n);
  uint64_t sum = 0;

  for (size_t d = 1; d * d <= g; ++d) {
    if (g % d == 0) {
      size_t d1 = d, d2 = g / d;

      {
        size_t nd = n / d1, md = m / d1;
        sum += (uint64_t)phi(d1) * binom(nd + md - 1, md - 1);
      }

      if (d1 != d2) {
        size_t nd = n / d2, md = m / d2;
        sum += (uint64_t)phi(d2) * binom(nd + md - 1, md - 1);
      }
    }
  }

  return sum / m;
}
