#include "bracelet.h"
#include "maths.h"
#include "mss.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

void bracelet_iter_new(bracelet_iter_t *it, size_t m, size_t tot,
                       size_t *scratch) {
  it->m = m;
  it->tot = tot;
  it->scratch = scratch;
  memset(it->scratch, 0, (m + 1) * sizeof(size_t));

  it->t = 1; // position
  it->p = 1; // period length
  it->r = 1;
  it->u = 1;
  it->v = 1;
  it->sum = 0;
  it->rs = 0;
  it->stage = ITER_STAGE_DESCEND;
}

static inline int check_rev(size_t *a, size_t t, size_t i) {
  for (size_t j = i + 1; j <= (t + 1) / 2; j++) {
    if (a[j] < a[t - j + 1])
      return 0;
    if (a[j] > a[t - j + 1])
      return -1;
  }
  return 1;
}

bool bracelet_iter_next(bracelet_iter_t *it, size_t *vec) {
  const size_t m = it->m;
  const size_t tot = it->tot;
  size_t *a = it->scratch;

  for (;;) {
    switch (it->stage) {
    case ITER_STAGE_DESCEND: {
      if (it->t > 1 && it->t - 1 > (m - it->r) / 2 + it->r) {
        if (a[it->t - 1] > a[m - it->t + 2 + it->r]) {
          it->rs = 0;
        } else if (a[it->t - 1] < a[m - it->t + 2 + it->r]) {
          it->rs = 1;
        }
      }

      if (it->t > m) { // leaf
        it->stage = ITER_STAGE_BACKTRACK;
        if (m % it->p == 0 && it->sum == tot) {
          vec[1] = a[m];
          memcpy(vec + 2, a + 1, (m - 1) * sizeof(size_t));
          size_t refl[2 * m];
          for (size_t i = 0; i < m; i++) {
            refl[i] = refl[i + m] = a[m - i];
          }

          size_t k = 0;
          for (size_t j = 1; j < 2 * m;) {
            size_t i = 0;
            while (i < m && refl[k + i] == refl[j + i])
              i++;

            if (i == m)
              break;

            if (refl[k + i] < refl[j + i]) {
              j = j + i + 1;
            } else {
              k = (j > k + i) ? j : k + i + 1;
              j = k + 1;
            }
          }
          size_t min_rot = k % m;

          int final_cmp = 0;
          for (size_t i = 0; i < m; i++) {
            if (a[i + 1] < refl[min_rot + i]) {
              final_cmp = -1;
              break;
            } else if (a[i + 1] > refl[min_rot + i]) {
              final_cmp = 1;
              break;
            }
          }

          if (final_cmp > 0)
            break;

          vec[0] = (final_cmp == 0) ? 0 : 1;
          return true;
        }
        break;
      }

      size_t v = a[it->t - it->p];
      if (it->sum + v <= tot) {
        a[it->t] = v;
        it->sum += v;

        if (v == a[1]) {
          it->v = it->v + 1;
        } else {
          it->v = 0;
        }

        if (it->u == it->t - 1 && it->t > 1 && a[it->t - 1] == a[1]) {
          it->u = it->u + 1;
        }

        if (it->t == m && it->u != m && a[m] == a[1]) {
          it->sum -= v;
          it->stage = ITER_STAGE_LOOP;
          a[it->t] = v + 1;
          break;
        }

        if (it->u == it->v) {
          int rev = check_rev(a, it->t, it->u);
          if (rev == -1) {
            it->sum -= v;
            it->stage = ITER_STAGE_LOOP;
            a[it->t] = v + 1;
            break;
          } else if (rev == 1) {
            it->r = it->t;
            it->rs = 0;
          }
        }

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

      a[it->t] = v;
      it->sum += v;

      if (it->t == 1) {
        it->p = 1;
        it->r = 1;
        it->u = 1;
        it->v = 1;
        it->rs = 0;
      } else {
        it->p = it->t;
        it->v = 0;
        it->rs = 0;
      }

      ++it->t;
      it->stage = ITER_STAGE_DESCEND;
      break;
    }

    case ITER_STAGE_BACKTRACK:
      --it->t;
      if (it->t == 0)
        return false;

      it->sum -= a[it->t];

      if (it->u == it->t) {
        it->u = it->u > 1 ? it->u - 1 : 1;
      }

      a[it->t] += 1;
      it->stage = ITER_STAGE_LOOP;
      break;
    }
  }
}
