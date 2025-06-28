#include "maths.h"
#include "mss.h"

#include <string.h>
#include <stdlib.h>

void mss_iter_new(mss_iter_t *const it, size_t n, size_t r, size_t *vec, size_t *scratch) {
  it->n = n;
  it->tot = r;
  it->vec = vec;
  it->scratch = scratch;

  it->lvl = 0;
  it->fin = false;
  memset(vec, 0, n*sizeof(size_t));
  memset(scratch, 0, n*sizeof(size_t));
}

void mss_iter_init_at(mss_iter_t *const it, size_t n, size_t r, size_t *vec, size_t *scratch) {
  it->n = n;
  it->tot = r;
  it->vec = vec;
  it->scratch = scratch;
  it->lvl = 1;
  it->fin = false;

  size_t rem = 0;
  for (size_t i = n; i-- > 0; ) {
    rem += vec[i];
    scratch[i] = rem;
  }
}

size_t mss_iter_size(size_t m, size_t n) {
  uint128_t numerator = 1;
  uint128_t denominator = 1;
  for (size_t i = 1; i < m; ++i) {
    int num = i + n - 1;
    int den = i;
    //greedily simplify the new terms
    if (num % den == 0) {
      numerator *= num/den;
    }
    else {
      numerator *= num;
      denominator *= den;
    }
    //greedily simplify the totals
    if (numerator % denominator == 0) {
      numerator = numerator / denominator;
      denominator = 1;
    }
  }
  return numerator / denominator;
}

bool mss_iter(mss_iter_t *restrict it) {
  if (it->fin) return false;

  const size_t n = it->n, tot = it->tot;
  size_t *vec = it->vec, *scratch = it->scratch;

  if (it->lvl == 0) {
    for (size_t i = 0; i < n - 1; ++i) {
      scratch[i] = (i == 0) ? tot : scratch[i-1] - vec[i-1];
      vec[i] = 0;
    }
    vec[n-1] = (n == 1) ? tot : scratch[n-2] - vec[n-2];
    it->lvl = n;
    return true;
  }

  for (int k = n - 2; k >= 0; --k) {
    size_t hi = (k == 0) ? tot : scratch[k-1] - vec[k-1];
    if (vec[k] < hi) {
      ++vec[k];

      for (size_t i = k + 1; i < n - 1; ++i) {
        scratch[i] = scratch[i-1] - vec[i-1];
        vec[i]     = 0;
      }
      vec[n-1] = scratch[n-2] - vec[n-2];
      return true;
    }
  }

  it->fin = true;
  return false;
}

void canon_iter_new(canon_iter_t *it, size_t m, size_t tot) {
  it->m = m;
  it->tot = tot;
  it->vec = malloc(m*sizeof(size_t));
  it->scratch = malloc((m+1)*sizeof(size_t));
  memset(it->scratch, 0, sizeof(size_t) * (m + 1));

  it->t = 1; // position
  it->p = 1; // period length
  it->sum = 0;
  it->stage = ITER_STAGE_DESCEND;
}

void canon_iter_free(canon_iter_t *it) {
  free(it->scratch);
  free(it->vec);
}

bool canon_iter_next(canon_iter_t *it) {
  const size_t m = it->m;
  const size_t tot = it->tot;
  size_t *a = it->scratch;

  for (;;) {
    switch (it->stage) {
    case ITER_STAGE_DESCEND: {
      if (it->t > m) { // leaf
        it->stage = ITER_STAGE_BACKTRACK;
        if (m % it->p == 0 && it->sum == tot) {
          for (size_t i = 0; i < m; ++i)
            it->vec[i] = a[i+1];
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
      a[it->t] = v+1;
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
        return false;

      it->sum -= a[it->t];
      a[it->t] += 1;
      it->stage = ITER_STAGE_LOOP;
      break;
    }
  }
}
