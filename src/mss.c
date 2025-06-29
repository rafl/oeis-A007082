#include "maths.h"
#include "mss.h"

#include <string.h>
#include <stdio.h>

void mss_iter_w_new(mss_iter_w_t *const it, size_t n, size_t m, size_t *vec, size_t *scratch) {
  it->n = n;
  it->m = m;
  it->vec = vec;
  
  it->pos = (size_t) 1;
  it->fin = false;

  //initialize with a zero-filled array with -1 in the final it->pos
  memset(vec, 0, m*sizeof(size_t));
  vec[0] = (size_t) -1;
  
  memset(scratch, 0, m*sizeof(size_t));
}

bool mss_iter_w(mss_iter_w_t *restrict it) {
  while (it->pos > 0) {
    ++it->vec[it->pos - 1];
    size_t prev = it->pos;
    //prev_length == m checks for satisfaction as lyndon word, sum() == n is a filter for oeis use case
    size_t sum = 0;
    for (size_t i = 0; i < it->m; ++i) {
      it->scratch[it->m-1-i] = it->vec[i];
      sum += it->vec[i];
    }
    printf("||%d||", sum);
    if (prev == (it->m) && sum == it->n) {
      return true;
    }
    //if it->pos is somewhere non-final, duplicate everything prior to it->pos into the cells after it->pos
    while (it->pos < it->m) {
      printf("--%d->", it->pos);
      it->vec[it->pos] = it->vec[it->pos - prev];
      ++it->pos;
    }
    //if the value at it->pos is the maximum value, move leftward
    while (it->pos > 0 && it->vec[it->pos-1] == it->n) {
      printf("<-%d--", it->pos);
      --it->pos;
    }
  }
  //all done if it->pos == 0 and no new term was already returned
  it->fin = true;
  return false;
}

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
