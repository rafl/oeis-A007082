#include "mss.h"
#include "debug.h"
#include "maths.h"

#include <stdlib.h>
#include <string.h>

void canon_iter_new(canon_iter_t *it, size_t m, size_t tot, mss_el_t *scratch,
                    size_t depth) {
  assert(depth >= 1 && depth <= m && tot < ((mss_el_t)-1));
  it->m = m;
  it->tot = tot;
  it->scratch = scratch;
  memset(it->scratch, 0, (m + 1) * sizeof(mss_el_t));
  it->scratch[0] = tot;

  it->t = 1; // position
  it->p = 1; // period length
  it->sum = 0;
  it->depth = depth;
  it->start_depth = 0;
  it->nonzero_count = 0;
  it->stage = ITER_STAGE_DESCEND;
}

// t p sum stage depth start_depth scratch...(m+1)
size_t canon_iter_save(canon_iter_t *it, void *buf, size_t len) {
  size_t n = (6 + it->m + 1) * sizeof(uint64_t);
  assert(len >= n);
  uint64_t *out = buf;
  memcpy(out,
         (uint64_t[]){it->t, it->p, it->sum, it->stage, it->depth,
                      it->start_depth},
         6 * sizeof(uint64_t));
  // write uint8s out as 64 bits. i don't wanna change the format again right
  // now
  for (size_t i = 0; i <= it->m; ++i)
    out[6 + i] = it->scratch[i];
  return n;
}

void canon_iter_resume(canon_iter_t *it, size_t m, size_t tot,
                       mss_el_t *scratch, const void *buf, size_t len) {
  assert(len >= (6 + m + 1) * sizeof(uint64_t));
  const uint64_t *in = buf;
  it->m = m;
  it->tot = tot;
  it->t = in[0];
  it->p = in[1];
  it->sum = in[2];
  it->stage = in[3];
  it->depth = in[4];
  it->start_depth = in[5];
  it->scratch = scratch;
  for (size_t i = 0; i <= m; ++i)
    it->scratch[i] = in[6 + i];

  // Compute nonzero_count from restored scratch buffer
  it->nonzero_count = 0;
  for (size_t i = 1; i < it->t; ++i) {
    if (it->scratch[i] != 0)
      it->nonzero_count++;
  }
}

bool canon_iter_next(canon_iter_t *it, mss_el_t *vec) {
  const size_t m = it->m;
  const size_t tot = it->tot;
  mss_el_t *a = it->scratch;

  for (;;) {
    switch (it->stage) {
    case ITER_STAGE_DESCEND: {
      if (it->t == it->depth + 1) {
        it->stage = ITER_STAGE_BACKTRACK;
        if (it->depth == m) {
          if (m % it->p == 0 && it->sum == tot) {
            memcpy(vec, a + 1, it->depth * sizeof(mss_el_t));
            return true;
          }
          break;
        }
        if (a[1] * (m - it->depth) >= (tot - it->sum)) {
          memcpy(vec, a + 1, it->depth * sizeof(mss_el_t));
          return true;
        }
        break;
      }

      if (it->t >= 2) {
        if (a[1] * (m - it->t + 1) < (tot - it->sum)) {
          it->stage = ITER_STAGE_BACKTRACK;
          break;
        }
      }

      size_t v = a[it->t - it->p];
      if (it->sum + v <= tot) {
        a[it->t] = v;
        it->sum += v;
        if (v != 0)
          it->nonzero_count++;
        ++it->t;
        break;
      }

      it->stage = ITER_STAGE_LOOP;
      a[it->t] = tot - it->sum;
      break;
    }

    case ITER_STAGE_LOOP: {
      size_t v = a[it->t];
      if (v > tot - it->sum) {
        it->stage = ITER_STAGE_BACKTRACK;
        break;
      }

      it->sum += v;
      if (v != 0)
        it->nonzero_count++;
      ++it->t;
      it->p = it->t - 1;
      it->stage = ITER_STAGE_DESCEND;
      break;
    }

    case ITER_STAGE_BACKTRACK:
      --it->t;
      if (it->t <= it->start_depth)
        return false;

      if (a[it->t] != 0)
        it->nonzero_count--;
      it->sum -= a[it->t];
      if (a[it->t] == 0) {
        break;
      }
      --a[it->t];
      it->stage = ITER_STAGE_LOOP;
      break;
    }
  }
}

void canon_iter_from_prefix(canon_iter_t *it, size_t m, size_t tot,
                            mss_el_t *scratch, const mss_el_t *prefix,
                            size_t prefix_depth) {
  assert(prefix_depth >= 1 && prefix_depth < m);
  it->m = m;
  it->tot = tot;
  it->scratch = scratch;

  it->scratch[0] = prefix[0];
  memcpy(it->scratch + 1, prefix, prefix_depth * sizeof(mss_el_t));
  memset(it->scratch + prefix_depth + 1, 0,
         (m - prefix_depth) * sizeof(mss_el_t));

  it->sum = 0;
  it->nonzero_count = 0;
  for (size_t i = 1; i <= prefix_depth; ++i) {
    it->sum += it->scratch[i];
    if (it->scratch[i] != 0)
      it->nonzero_count++;
  }

  it->t = prefix_depth + 1;

  it->p = 1;
  for (size_t p = 1; p <= prefix_depth; ++p) {
    bool matches = true;
    for (size_t i = p + 1; i <= prefix_depth; ++i) {
      if (it->scratch[i] != it->scratch[i - p]) {
        matches = false;
        break;
      }
    }
    if (matches) {
      it->p = p;
      break;
    }
  }

  it->depth = m;
  it->start_depth = prefix_depth;
  it->stage = ITER_STAGE_DESCEND;
}

// not tuned well yet
size_t canon_iter_depth_for(size_t m) { return (m - 1) / 2; }

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
