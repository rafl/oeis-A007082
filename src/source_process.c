#include "debug.h"
#include "source_process.h"
#include "maths.h"
#include "primes.h"
#include "mss.h"
#include "progress.h"
#include "queue.h"
#include "snapshot.h"

#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>
#include <stdio.h>

static uint64_t m_for(uint64_t n) {
  return 2*((n+1)/4)+1;
}

static void create_exps(size_t *ms, // "necklace normalzed" coefficients
  size_t len, 
  uint64_t *dst) {
  size_t idx = 0;

  for (size_t exp = 0; exp < len; ++exp) {
    // number of times we're repeating a given coeff is ms[exp] - unless exp == for some reasons
    // and then we subtract 1?
    // Oh because we always have 1 as one of the args? Hmmm ok
    // I guess this is why we skip the zero case too?
    size_t reps = ms[exp] - (exp == 0);
    for (size_t k = 0; k < reps; ++k)
      dst[idx++] = exp; // So we're turning 1 3 7 into 1 2 2 2 3 3 3 3 3 3 3 ( except for the skip the first 1 thing)
  }

  dst[idx] = 0;
}

// precomputation of shared forms
// "montgomery form" (do some bitshifts and then a mult instead of a divide)
// That's in maths.c
typedef struct {
  uint64_t n, m, p, //n = number of veritcies //obvious
  p_dash, r, r2, // montgomery stuff _M implies something is in montgomery form
  *jk_prod_M, // cache of w^j*w^-k / (w^-j*w^k + w^j*w^-k)
  *nat_M, // natural numbers up to n (inclusive)
  *nat_inv_M, // inverses of natural numbers up to n (inclusive)
  *jk_sums_M, // w^-j*w^k + w^j*w^-k 
  *ws_M, // powers of omega (m form)
  *fact_M, // i! for i <= n
  *fact_inv_M; // 1/i! for i <= n
} prim_ctx_t;

// index into rectanular array
static inline size_t jk_pos(size_t j, size_t k, uint64_t m) {
  return j*m + k;
}

// shared over threads - not mutated
static prim_ctx_t *prim_ctx_new(uint64_t n, uint64_t m, uint64_t p, uint64_t w) {
  prim_ctx_t *ctx = malloc(sizeof(prim_ctx_t));
  assert(ctx);
  ctx->n = n;
  ctx->m = m;
  ctx->p = p;
  ctx->p_dash = (uint64_t)(-inv64_u64(p));
  ctx->r = ((uint128_t)1 << 64) % p;
  ctx->r2 = (uint128_t)ctx->r * ctx->r % p;

  // initialize roots of unity
  ctx->ws_M = malloc(m*sizeof(uint64_t));
  assert(ctx->ws_M);
  ctx->ws_M[0] = ctx->r;
  ctx->ws_M[1] = mont_mul(w, ctx->r2, p, ctx->p_dash);
  for (size_t i = 2; i < m; ++i)
    ctx->ws_M[i] = mont_mul(ctx->ws_M[i-1], ctx->ws_M[1], p, ctx->p_dash);

  // w^j * w^-k lookup - not actually inserted into the context
  uint64_t jk_pairs_M[m*m];
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      jk_pairs_M[jk_pos(j, k, m)] = mont_mul(ctx->ws_M[j], ctx->ws_M[k ? m-k : 0], p, ctx->p_dash);
  }

  // cache of // w^-j*w^k + w^j*w^-k 
  ctx->jk_sums_M = malloc(m*m*sizeof(uint64_t));
  assert(ctx->jk_sums_M);
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      ctx->jk_sums_M[jk_pos(j, k, m)] =
        add_mod_u64(jk_pairs_M[jk_pos(j, k, m)], jk_pairs_M[jk_pos(k, j, m)], p);
  }

  // cache of w^j*w^-k / (w^-j*w^k + w^j*w^-k)
  ctx->jk_prod_M = malloc(m*m*sizeof(uint64_t));
  assert(ctx->jk_prod_M);
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k) {
      size_t pos = jk_pos(j, k, m);
      uint64_t sum_inv = mont_inv(ctx->jk_sums_M[pos], ctx->r, p, ctx->p_dash);
      ctx->jk_prod_M[pos] = mont_mul(jk_pairs_M[pos], sum_inv, p, ctx->p_dash);
    }
  }

  // 1 to n
  ctx->nat_M = malloc((n+1)*sizeof(uint64_t));
  assert(ctx->nat_M);
  for (size_t i = 0; i <= n; ++i)
    ctx->nat_M[i] = mont_mul(i, ctx->r2, p, ctx->p_dash);

  // 1/i for i = 1 to n
  ctx->nat_inv_M = malloc((n + 1) * sizeof(uint64_t));
  assert(ctx->nat_inv_M);
  ctx->nat_inv_M[0] = 0;
  for (size_t k = 1; k <= n; ++k)
    ctx->nat_inv_M[k] = mont_inv(ctx->nat_M[k], ctx->r, p, ctx->p_dash);

  // i! for i = 1 to n
  ctx->fact_M = malloc(n*sizeof(uint64_t));
  assert(ctx->fact_M);
  ctx->fact_M[0] = ctx->r;
  for (size_t i = 1; i < n; ++i)
    ctx->fact_M[i] = mont_mul(ctx->fact_M[i-1], ctx->nat_M[i], p, ctx->p_dash);

  // 1/i! for i = 1 to n
  ctx->fact_inv_M = malloc(n*sizeof(uint64_t));
  assert(ctx->fact_inv_M);
  ctx->fact_inv_M[n-1] = mont_inv(ctx->fact_M[n-1], ctx->r, p, ctx->p_dash);
  for (size_t i = n-1; i; --i)
    ctx->fact_inv_M[i-1] = mont_mul(ctx->fact_inv_M[i], ctx->nat_M[i], p, ctx->p_dash);

  return ctx;
}

static void prim_ctx_free(prim_ctx_t *ctx) {
  free(ctx->fact_inv_M);
  free(ctx->fact_M);
  free(ctx->ws_M);
  free(ctx->jk_sums_M);
  free(ctx->nat_inv_M);
  free(ctx->nat_M);
  free(ctx->jk_prod_M);
  free(ctx);
}

static void build_drop_mat(uint64_t *A, size_t dim, uint64_t *exps, prim_ctx_t *ctx) {
  const size_t r = dim+1;
  const uint64_t p = ctx->p, m = ctx->m;

  for (size_t j = 0; j < dim; ++j) {
    uint64_t acc = 0;

    for (size_t k = 0; k < r; ++k) if (j != k) {
      const size_t pos = jk_pos(exps[j], exps[k], m);
      uint64_t t = ctx->jk_prod_M[pos];

      acc = add_mod_u64(acc, t, p);

      if (k < dim) {
        A[j*dim + k] = (t == 0) ? 0 : p - t;
      }
    }

    A[j*dim + j] = acc;
  }

  for (int i = 0; i < dim* dim; i++)
  {
    printf("%lu, ", A[i]);
  }

  printf("\n");
}


// Calculate the multinomal coefficient where the powers of x_i are given by *ms
static uint64_t multinomial_mod_p(const prim_ctx_t *ctx, const size_t *ms, size_t len) {
  const uint64_t p = ctx->p, p_dash = ctx->p_dash;

  size_t tot = 0;
  // is this total not just n - 1?
  for (size_t i = 0; i < len; ++i)
    tot += ms[i] - (i == 0);

  uint64_t coeff = ctx->fact_M[tot];
  for (size_t i = 0; i < len; ++i)
    // - (i==0) is to subtract 1 from the w^0 coeff  - this is why we can't only calc this once
    coeff = mont_mul(coeff, ctx->fact_inv_M[ms[i] - (i == 0)], p, p_dash);

  return coeff;
}

// Compute the determinant via gaussian elimination
// "we carry a factor through rather than converting all the slow inverses"
static uint64_t det_regular(uint64_t *A, size_t dim) {
  // const uint64_t p = ctx->p, p_dash = ctx->p_dash;
  // uint64_t det = ctx->r, scaling_factor = ctx->r;
  uint64_t det = 1, scaling_factor = 1;

  // we're saying the thing multiplied by dim is the row index
  // for each pivot "column"
  for (size_t k = 0; k < dim; ++k) {
    // start with pivot "row" being the same as col
    size_t pivot_i = k;
    // skip over rows if the cell is zero
    while (pivot_i < dim && A[pivot_i*dim + k] == 0) ++pivot_i;
    // if there was non zero cell - det is zero
    if (pivot_i == dim) return 0;
    // if we did skip over some cells

    // This really just handles the case where a zero is on the diagonal
    // not used much besides
    if (pivot_i != k) {
      // we're going to swap two sub... rows? - using k
      for (size_t j = 0; j < dim; ++j) {
        uint64_t tmp = A[k*dim + j];
        A[k*dim + j] = A[pivot_i*dim + j];
        A[pivot_i*dim + j] = tmp;
      }
      det = - det; // flip this sign
    }

    uint64_t pivot = A[k*dim + k];
    // multiply in our "pivot" point value
    // det = mont_mul(det, pivot, p, p_dash);
    det = det * pivot;

    // This is the actual subtraction logic
    for (size_t i = k + 1; i < dim; ++i) {
      // scaling_factor = mont_mul(scaling_factor, pivot, p, p_dash);
      scaling_factor = scaling_factor * pivot;
      uint64_t multiplier = A[i*dim + k];
      for (size_t j = k; j < dim; ++j)
      // mul and subtract off the rest
        // A[i*dim + j] = mont_mul_sub(A[i*dim + j], pivot, A[k*dim + j], multiplier, p, p_dash);
        A[i*dim + j] = (A[i*dim + j] * pivot)-(A[k*dim + j] * multiplier);
    }
  }

  return (int64_t)(det) / (int64_t)scaling_factor;    //mont_mul(det, mont_inv(scaling_factor, ctx->r, p, p_dash), p, p_dash);
}

// Compute the determinant via gaussian elimination
// "we carry a factor through rather than converting all the slow inverses"
static uint64_t det_mod_p(uint64_t *A, size_t dim, const prim_ctx_t *ctx) {
  for (int i = 0; i < 9; i++)
  {
    // printf("input %lu \n", mont_mul(A[i], 1, ctx->p, ctx->p_dash));
  }

  const uint64_t p = ctx->p, p_dash = ctx->p_dash;
  uint64_t det = ctx->r, scaling_factor = ctx->r;

  // we're saying the thing multiplied by dim is the row index
  // for each pivot "column"
  for (size_t k = 0; k < dim; ++k) {
    // start with pivot "row" being the same as col
    size_t pivot_i = k;
    // skip over rows if the cell is zero
    while (pivot_i < dim && A[pivot_i*dim + k] == 0) ++pivot_i;
    // if there was non zero cell - det is zero
    if (pivot_i == dim) return 0;
    // if we did skip over some cells

    for (int i = 0; i < 9; i++)
    {
      int64_t val = mont_mul(A[i], 1, ctx->p, ctx->p_dash);
      if (val > p / 2) val = val - p;
      // printf("%li, ", val);
    }
    // printf("\n");

    // This really just handles the case where a zero is on the diagonal
    // not used much besides
    if (pivot_i != k) {
      // we're going to swap two sub... rows? - using k
      printf("zero on diagonal\n");
      for (size_t j = 0; j < dim; ++j) {
        uint64_t tmp = A[k*dim + j];
        A[k*dim + j] = A[pivot_i*dim + j];
        A[pivot_i*dim + j] = tmp;
      }
      det = p - det; // flip this sign
    }

    uint64_t pivot = A[k*dim + k];
    // multiply in our "pivot" point value
    det = mont_mul(det, pivot, p, p_dash);

    // This is the actual subtraction logic
    for (size_t i = k + 1; i < dim; ++i) {
      scaling_factor = mont_mul(scaling_factor, pivot, p, p_dash);
      uint64_t multiplier = A[i*dim + k];
      for (size_t j = k; j < dim; ++j)
      // mul and subtract off the rest
        A[i*dim + j] = mont_mul_sub(A[i*dim + j], pivot, A[k*dim + j], multiplier, p, p_dash);
    }
  }

  int64_t val =  mont_mul(det, 1, ctx->p, ctx->p_dash);
  if (val > p / 2) val = val - p;
  // printf("inner_det %li \n", val);

  val =  mont_mul(scaling_factor, 1, ctx->p, ctx->p_dash);
  if (val > p / 2) val = val - p;
  // printf("scale_factor %li \n", val);

  int64_t invTerm = mont_inv(scaling_factor, ctx->r, p, p_dash);

  return mont_mul(det, invTerm, p, p_dash);
}

// I think that we could compute this for the *vect form of this if we liked
// The full product over all pairwise combinations
static uint64_t f_fst_term(uint64_t *exps, const prim_ctx_t *ctx) {
  uint64_t acc = ctx->r; // r is basically 1 in mont form
  for (size_t j = 0; j < ctx->n; ++j) {
    for (size_t k = j + 1; k < ctx->n; ++k) {
      uint64_t t = ctx->jk_sums_M[jk_pos(exps[j], exps[k], ctx->m)];
      acc = mont_mul(acc, t, ctx->p, ctx->p_dash);
    }
  }
  return acc;
}

static uint64_t f_snd_trm(uint64_t *c, const prim_ctx_t *ctx) {
  const uint64_t p = ctx->p, m = ctx->m;

  // active groups
  // This is the indexs of the non zero elements of c
  // Which is also the powers of w they correspond to
  // typ for type / class / colour
  size_t typ[m], r = 0;
  for (size_t i = 0; i < m; ++i) {
    if (c[i]) {
      typ[r] = i;
      ++r;
    }
  }

  uint64_t prod_M = ctx->r;
  
  // for each non zero power of omega w^i in our args
  for (size_t a = 0; a < r; ++a) {
    size_t i = typ[a];

    // This is basically the column sum
    uint64_t sum = 0;
    // for each non zero power of omega w^j in our args
    for (size_t b = 0; b < r; ++b) {
      size_t j = typ[b];
      // w^i*w^-j
      // (n.b. we could distribute over w^i here if we wanted)

      // This is the inner element of A matrix from the paper (up to sign) / inner element of product sum in paper
      uint64_t w = ctx->jk_prod_M[jk_pos(i, j, m)];

      // sum += w * multiplicity of (w^j)
      sum = add_mod_u64(sum, mont_mul(ctx->nat_M[c[j]], w, p, ctx->p_dash), p);

      // let B_jk = w^j*w^-k / (w^-j*w^k + w^j*w^-k)
      // reorder args to f such they increase in order of power of omega
      // Then we have (x_0, x_1, x_2...)
      // Let B_kl = x_k*x_l^-1 / (.....)
      // then sum(k) = sum over l of B_kl
    }

    prod_M = mont_mul(prod_M, mont_pow(sum, c[i]-1, ctx->r, p, ctx->p_dash), p, ctx->p_dash);

    // prod_M - product over each deleted row / col (we're leaving 1 behind) 
    // of the sum over l of B_kl
    // So like for each column we "delete" we multiply by the column sum...
  }

  // #### Up to here prod_M is the same
  // Prod M = product[a = 0->r-1] sum[b = 0->1-r] w^coeff_cnt[a] * w^-coeff_cnt[b]

  // #### And here is the point is diverges
  prod_M = mont_mul(prod_M, ctx->nat_inv_M[c[0]], p, ctx->p_dash);

  // Put this isn't the only difference


  // We divide by the multiplicty of 1??

  // quotient minor
  size_t dim = r - 1;
  // If all terms were the same power of w and we quotiented everything out
  if (!dim)
    return prod_M;

  // We're constructing the minor of the reduced matrix
  // dropping the first row / col
  uint64_t A[dim*dim];
  for (size_t a = 1; a < r; ++a) {
    size_t i = typ[a];

    // contribution from the deleted block?????
    uint64_t w_del = ctx->jk_prod_M[jk_pos(i, 0, m)];

    // When making the "laplaican" the diagonal is the sum of all off-diagonal elements in the row of the full matrix
    // including the column dropped to form the minor. So we special add that to the diag here (with multiplicity)
    uint64_t diag = mont_mul(ctx->nat_M[c[0]], w_del, p, ctx->p_dash);

    // remaining off-diagonal blocks
    for (size_t b = 1; b < r; ++b) {
      size_t j = typ[b];
      if (j == i)
        continue;

      uint64_t w = ctx->jk_prod_M[jk_pos(i, j, m)];

      // Again fill the matrix as per it's normal terms but with multiplicity
      uint64_t v = mont_mul(ctx->nat_M[c[j]], w, p, ctx->p_dash);

      // This is the -1 coefficient on the off diag elements
      A[(a-1)*dim + (b-1)] = p - v;
      // and add it to the total for the diag elements
      diag = add_mod_u64(diag, v, p);
    }

    A[(a-1)*dim + (a-1)] = diag;
  }

  // I guess prod_M is some magic compensation coefficient
  return mont_mul(prod_M, det_mod_p(A, dim, ctx), p, ctx->p_dash);
}

// Exps = exponents of powers of unity [0, 1, 1, 2, 4] -> [1, w, w, w^2, w^4]
// Vec is staying in count space (so would be ) [1, 2, 1, 0, 1] for above
static uint64_t f(uint64_t *vec, uint64_t *exps, const prim_ctx_t *ctx) {
  return mont_mul(f_fst_term(exps, ctx), f_snd_trm(vec, ctx), ctx->p, ctx->p_dash);
}

uint64_t f_snd_trm_old(uint64_t *vec, uint64_t *exps, prim_ctx_t *ctx) {
  size_t dim = ctx->n-1;
  uint64_t A[dim*dim];

  build_drop_mat(A, dim, exps, ctx);

  for (int i = 0; i < dim*dim; i++)
  {
    // printf("drop_mat %lu \n", mont_mul(A[i], 1, ctx->p, ctx->p_dash));
  }
  // printf("end_drop_mat\n");

  // return mont_mul(det_mod_p(A, dim, ctx), 1, ctx->p, ctx->p_dash);
  return det_mod_p(A, dim, ctx);
}

// state of whole process (shared over threads)
typedef struct {
  uint64_t n, /* element of seq */ m, /*w is the mth root of unity*/ *ps /* list of primes */;
  size_t idx /* withth prime*/, np /* number of primes*/, *vecss /*Some buffers for vectors*/;
  bool quiet, snapshot; /*mode stuff - saving snapshots so you can restart*/
  size_t n_thrds; /*number of threads*/
} proc_state_t;

typedef struct {
  _Atomic size_t *done;
  const prim_ctx_t *ctx;
  queue_t *q;
  size_t *vecs;
  uint64_t *l_acc, *acc;
  pthread_mutex_t *acc_mu;
  bool idle;
} worker_t;

static void w_resume(void *ud) {
  worker_t *w = ud;
  w->idle = false;
}

static resume_cb_t w_idle(void *ud) {
  worker_t *w = ud;
  pthread_mutex_lock(w->acc_mu);
  *w->acc = add_mod_u64(*w->acc, *w->l_acc, w->ctx->p);
  *w->l_acc = 0;
  pthread_mutex_unlock(w->acc_mu);
  w->idle = true;
  return w_resume;
}

// thread function for doing work on the "maths stuff"
static void *residue_for_prime(void *ud) {
  worker_t *worker = ud;
  const prim_ctx_t *ctx = worker->ctx;
  uint64_t n = ctx->n, m = ctx->m, p = ctx->p;
  uint64_t exps[n], /* what are you?*/ l_acc = 0; // l_acc is where we're going to accumulate the total residual from the stuff we pull from our work queue
  size_t *vecs = worker->vecs;
  worker->l_acc = &l_acc;

  for (;;) {
    size_t n_vec = queue_pop(worker->q, vecs, w_idle, worker);
    if (!n_vec) break;

    for (size_t c = 0; c < n_vec; ++c) {
      // each vector has len m
      // each vector contains "necklace normalized" coefficient counts
      // i.e. 1 3 7 = 1 lot of w^0, 3 lots of w^1, 7 lots of w^2
      size_t *vec = &vecs[c*m];
      create_exps(vec, m, exps);
      // This is the full f from the paper
      // We evaluate it in two halfs (the product and the matrix determinant from theorem 2)
      uint64_t f_0 = f(vec, exps, ctx);
      size_t vec_rots[2*m];
      memcpy(vec_rots, vec, m*sizeof(uint64_t));
      memcpy(vec_rots+m, vec_rots, m*sizeof(uint64_t));
      // go over each "rotation" of the vector - this is the same as multiply by omega
      // e.g. go from f_0 = f(1 3 7 0) to f_1(0 1 3 7) is the equivient of multiplying each coeffecient
      // by omega
      // we index in reverse to make this the same as multiplying by w
      for (size_t r = 0; r < m; ++r) {
        size_t *vec_r = vec_rots + r;
        // We can skip this because... I am not sure
        // Because we always have at least one 1 in our calc I guess...
        if (vec_r[0] == 0) continue;
        // TODO - why do we compute this for each loop - isn't this a constant for a given set of coeffs?
        // Are we messing around with this a bit because of the constant 1 at the end? (yes - subtract one from w^0 term)
        uint64_t coeff = multinomial_mod_p(ctx, vec_r, m);
        size_t idx = (2*r) % m;
        // coeff * f_0 * w^(m-idx) = coeff * f_0 * w^-2r
        uint64_t f_n = mont_mul(coeff, mont_mul(f_0, ctx->ws_M[idx ? m-idx : 0], p, ctx->p_dash), p, ctx->p_dash);
        l_acc = add_mod_u64(l_acc, f_n, p);
      }
    }
    atomic_fetch_add_explicit(worker->done, n_vec, memory_order_relaxed);
  }

  (void)w_idle(worker);
  return NULL;
}

static size_t get_num_threads() {
  char *env = getenv("OMP_NUM_THREADS");
  if (env) {
    char *endptr;
    long val = strtol(env, &endptr, 10);
    if (*endptr == '\0' && val > 0)
      return (size_t)val;
  }

  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return n > 0 ? (size_t)n : 1;
}

static int ret(proc_state_t *st, prim_ctx_t *ctx, uint64_t acc, uint64_t *res, uint64_t *p_ret) {
  uint64_t denom = mont_inv(mont_pow(ctx->nat_M[ctx->m], ctx->n-1, ctx->r, ctx->p, ctx->p_dash), ctx->r, ctx->p, ctx->p_dash);
  uint64_t ret = mont_mul(mont_mul(acc, denom, ctx->p, ctx->p_dash), 1, ctx->p, ctx->p_dash);
  prim_ctx_free(ctx);

  *p_ret = st->ps[st->idx++];
  *res = ret;
  return 1;
}


// // TODO: move back into montgomery domain?
// uint64_t f_snd_trm_fast(uint64_t *vec, uint64_t *exps, prim_ctx_t *ctx) {
//   const uint64_t p = ctx->p, m = ctx->m;

//   size_t c[m];
//   // TODO: avoid copy? could adjust vec representation or handle vec[0] special below
//   memcpy(c, vec, m*sizeof(size_t));
//   ++c[0];

//   // active groups
//   size_t typ[m], r = 0, del_i = (size_t)-1;
//   for (size_t i = 0; i < m; ++i) {
//     if (c[i]) {
//       typ[r] = i;
//       // delete first non-empty
//       if (del_i == (size_t)-1)
//         del_i = i;
//       ++r;
//     }
//   }
//   const uint64_t c_del = c[del_i];

//   uint64_t prod_int = 1;
//   for (size_t a = 0; a < r; ++a) {
//     size_t i = typ[a];
//     uint64_t sum = 0;
//     for (size_t b = 0; b < r; ++b) {
//       size_t j = typ[b];
//       uint64_t w = ctx->jk_prod[jk_pos(i, j, m)];
//       sum = add_mod_u64(sum, mul_mod_u64(c[j], w, p), p);
//     }
//     uint64_t d_i = sub_mod_u64(sum, ctx->jk_prod[jk_pos(i, i, m)], p);
//     uint64_t lam_i = add_mod_u64(d_i, ctx->jk_prod[jk_pos(i, i, m)], p);

//     if (c[i] > 1)
//       prod_int = mul_mod_u64(prod_int, pow_mod_u64(lam_i, c[i] - 1, p), p);
//   }

//   // quotient minor
//   size_t dim = r - 1;
//   uint64_t A[dim ? dim*dim : 1];

//   if (dim) {
//     size_t row = 0;
//     for (size_t a = 0; a < r; ++a) {
//       size_t i = typ[a];
//       if (i == del_i) continue;

//       uint64_t diag = 0;
//       size_t col  = 0;

//       // contribution from the deleted block
//       uint64_t w_del = ctx->jk_prod[jk_pos(i, del_i, m)];
//       uint64_t val = mul_mod_u64(c[del_i], w_del, p);
//       diag = add_mod_u64(diag, val, p);

//       // remaining off-diagonal blocks
//       for (size_t b = 0; b < r; ++b) {
//         size_t j = typ[b];
//         if (j == del_i)
//           continue;

//         if (j == i)
//           continue;

//         if (col == row)
//           ++col;

//         uint64_t w  = ctx->jk_prod[jk_pos(i, j, m)];
//         uint64_t v  = mul_mod_u64(c[j], w, p);

//         A[row*dim + col] = v ? p - v : 0;
//         diag = add_mod_u64(diag, v, p);
//         ++col;
//       }

//       A[row*dim + row] = diag;
//       ++row;
//     }
//   }
//   for (size_t i = 0; i < dim*dim; ++i)
//     A[i] = mont_mul(A[i], ctx->r2, p, ctx->p_dash);
//   uint64_t det_q = dim ? mont_mul(det_mod_p(A, dim, ctx), 1, p, ctx->p_dash) : 1;
//   det_q = mul_mod_u64(det_q, inv_mod_u64(c_del, p), p);
//   return mul_mod_u64(prod_int, det_q, p);
// }

// Implementation of the virtual function - entrypoint for calculation
static int proc_next(source_t *self, uint64_t *res, uint64_t *p_ret) {
  proc_state_t *st = self->state;
  if (st->idx == st->np) return 0;

  // n, m, prime
  uint64_t n = st->n, m = st->m, p = st->ps[st->idx];
  // root of unity
  uint64_t w = mth_root_mod_p(p, m);
  prim_ctx_t *ctx = prim_ctx_new(n, m, p, w);

  const size_t siz = canon_iter_size(m, n);

  _Atomic size_t done = 0;
  uint64_t acc = 0;

  uint64_t iter_st[m+5];
  size_t st_len = 0;

  if (st->snapshot)
    snapshot_try_resume(n, p, &done, &acc, iter_st, &st_len);
  assert(done <= siz);

  if (done == siz) return ret(st, ctx, acc, res, p_ret);

  progress_t prog;
  if (!st->quiet)
  // progress bar stuff
    progress_start(&prog, p, &done, siz);

  // shared work queue
  queue_t *q = queue_new(n, m, iter_st, st_len, &st->vecss[st->n_thrds*CHUNK*m]);

  // make some threads
  pthread_t worker[st->n_thrds];
  worker_t w_ctxs[st->n_thrds];
  bool *idles[st->n_thrds];
  pthread_mutex_t acc_mu = PTHREAD_MUTEX_INITIALIZER;
  for (size_t i = 0; i < st->n_thrds; ++i) {
    w_ctxs[i] = (worker_t){ .ctx = ctx, .done = &done, .q = q, .vecs = &st->vecss[i*CHUNK*m], .idle = false, .acc = &acc, .acc_mu = &acc_mu };
    idles[i] = &w_ctxs[i].idle;
    // worker threads
    pthread_create(&worker[i], NULL, residue_for_prime, &w_ctxs[i]);
  }

  snapshot_t ss;
  if (st->snapshot)
    snapshot_start(&ss, n, p, st->n_thrds, q, idles, &done, &acc);

  queue_fill(q);

  for (size_t i = 0; i < st->n_thrds; ++i)
    pthread_join(worker[i], NULL);

  queue_free(q);
  if (st->snapshot)
    snapshot_stop(&ss);

  if (!st->quiet)
    progress_stop(&prog);

  return ret(st, ctx, acc, res, p_ret);
}

static void proc_destroy(source_t *self) {
  proc_state_t *st = self->state;
  free(st->vecss);
  free(st->ps);
  free(st);
  free(self);
}

#define P_STRIDE (1ULL << 10)

source_t *source_process_new(uint64_t n, uint64_t m_id, bool quiet, bool snapshot) {
  uint64_t m = m_for(n);
  size_t np;
  assert(m_id < P_STRIDE);
  uint64_t *ps = build_prime_list(n, m, m_id, P_STRIDE, &np);

  proc_state_t *st = malloc(sizeof(*st));
  assert(st);
  size_t n_thrds = get_num_threads();
  size_t *vecss = malloc(CHUNK*m*(n_thrds+1+Q_CAP)*sizeof(size_t));
  assert(vecss);
  *st = (proc_state_t){ .n = n, .m = m, .idx = 0, .np = np, .ps = ps, .quiet = quiet, .snapshot = snapshot, .n_thrds = n_thrds, .vecss = vecss };

  source_t *src = malloc(sizeof *src);
  assert(src);
  *src = (source_t){ .next = proc_next, .destroy = proc_destroy, .state = st };
  return src;
}

typedef struct {
  uint64_t value;
  uint64_t index;
} root_of_unity_t;


void print_summary(size_t val, root_of_unity_t const * roots, size_t m)
{
  printf("value = %lu that is:", val);
  for (size_t i = 0; i < m; i++)
  {
    printf("%lu*w^%lu + ", val/roots[i].value, roots[i].index);
    val %= roots[i].value;
  }
  printf("\n");
}

int comp_root(const void * e1, const void * e2)
{
  root_of_unity_t* elem1 = (root_of_unity_t*) e1;
  root_of_unity_t* elem2 = (root_of_unity_t*) e2;
  if (elem1->value < elem2->value) return 1;
  if (elem1->value > elem2->value) return -1;
  return 0;
}

void jack_test()
{
    printf("hello world!\n");

  size_t n = 5;
  size_t m = 5;
  // size_t p = 7;
  
  size_t vec[] = {1, 2, 2, 0, 0}; // sum should be n? - len should be m

  size_t np;

  size_t* primes = build_prime_list(n, m, 1, P_STRIDE, &np);

  size_t p = *primes;
  size_t const w = mth_root_mod_p(p, m);

  root_of_unity_t roots[n];
  uint64_t cur = 1;
  for (size_t i = 0; i < m; i++)
  {
    roots[i].index = i;
    roots[i].value = cur;
    // printf("w^%lu=%lu\n", i, cur);
    cur = mul_mod_u64(cur, w, p);
  }
  printf("w^%lu=%lu\n", m, cur);

  qsort(roots, n, sizeof(root_of_unity_t), comp_root);

  printf("p=%lu, w=%lu\n", p, w);

  prim_ctx_t *ctx = prim_ctx_new(n, m, p, w);

  // size_t mat[] = {1,2,3,7,8,10,4,6,6};
  // size_t mat_m[] = {1,2,3,7,8,10,4,6,6};

  // size_t basicDet = det_regular(mat, 3);
  // printf("determinant basic %lu\n", basicDet);

  // // mat = {1,2,3,7,8,10,4,6,6};

  // for (size_t i = 0; i < 9; i++)
  // {
  //   mat_m[i] = mont_mul(mat_m[i], ctx->r2, p, ctx->p_dash);
  //   printf("testing %lu \n", mont_mul(mat_m[i], 1, p, ctx->p_dash));
  // }
  
  // size_t det = det_mod_p(mat_m, 3, ctx);

  // printf("determinant %lu\n", mont_mul(det, 1, ctx->p, ctx->p_dash));

  // return;

  uint64_t vec_rots[2*m];
  // uint64_t exps[n];
  memcpy(vec_rots, vec, m*sizeof(uint64_t));
  memcpy(vec_rots+m, vec, m*sizeof(uint64_t));
  

  
  size_t f_2_0 = f_snd_trm(vec_rots, ctx);

  // int64_t calcedExps[n];

  int64_t exps[] = {
    0, 1, 1, 2, 2,
    4, 0, 0, 1, 1,
    3, 4, 4, 0, 0,
    2, 3, 3, 4, 4,
    1, 2, 2, 3, 3,
    };

  int64_t exps_perms[] = {
    1, 1, 2, 2, 0,
    4, 0, 1, 1, 0,
    3, 4, 4, 0, 0,
    2, 3, 3, 4, 4,
    1, 2, 2, 3, 3,
    };

  for (size_t i = 0; i < m; i++)
  {
    if (vec_rots[i] == 0) continue;
    // create_exps(vec_rots+i, m, calcedExps);

    printf("Vec rot\n");
    for (size_t k = 0; k < n; k++)
    {
      printf("%lu, ", (vec_rots+i)[k]);
    }
    printf("\n");


    printf("Hardcoded version\n");
    for (size_t k = 0; k < n; k++)
    {
      printf("%lu, ", (exps+5*i)[k]);
    }
    printf("\n");

    // printf("Calced version\n");
    // // for (size_t k = 0; k < n; k++)
    // // {
    // //   printf("%lu, ", calcedExps[k]);
    // // }
    // printf("\n");


    size_t f_first_m  = f_fst_term(exps+5*i, ctx);

    size_t f_second_m = f_snd_trm(vec_rots+i, ctx);
    size_t f_full = mont_mul(f_first_m, f_second_m, ctx->p, ctx->p_dash);
    size_t old_hard = f_snd_trm_old(vec_rots+i, exps+5*i, ctx);
    size_t old_perms = f_snd_trm_old(vec_rots+i, exps_perms+5*i, ctx);
    // size_t old_calced = f_snd_trm_old(vec_rots+i, calcedExps, ctx);

    size_t idx = (2*i) % m;
    // coeff * f_0 * w^(m-idx) = coeff * f_0 * w^-2r
    uint64_t f_2_n = mont_mul(f_2_0, ctx->ws_M[idx ? m-idx : 0], p, ctx->p_dash);


    //  printf("first factor %lu\n", mont_mul(f_first_m, 1, ctx->p, ctx->p_dash));
     printf("second factor  %lu\n", mont_mul(f_second_m, 1, ctx->p, ctx->p_dash));
     printf("second factor old hard %lu\n", mont_mul(old_hard, 1, ctx->p, ctx->p_dash));
     printf("second factor old permed %lu\n", mont_mul(old_perms, 1, ctx->p, ctx->p_dash));
    //  printf("second factor old calced %lu\n", mont_mul(old_calced, 1, ctx->p, ctx->p_dash));
     printf("second dave factor  %lu\n", mont_mul(f_2_n, 1, ctx->p, ctx->p_dash));

     printf("seconds hard raw permed (target) %lu\n", old_perms);
    for (size_t i = 0; i <= m; i++)
    {
      printf("seconds hard raw * w^%lu: %lu\n", i, mont_mul(old_hard, ctx->ws_M[i], ctx->p, ctx->p_dash));
      // ctx->ws_M[0]
    }

     printf("\n");
    // printf("overall  %lu\n", mont_mul(f_full, 1, ctx->p, ctx->p_dash));
  }
}