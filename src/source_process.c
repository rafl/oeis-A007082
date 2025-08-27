#include "debug.h"
#include "source_process.h"
#include "maths.h"
#include "primes.h"
#include "mss.h"
#include "progress.h"
#include "queue.h"
#include "snapshot.h"
#include "cyclotomic_field.h"

#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <unistd.h>
#include <stdio.h>

using FieldT = CyclomaticFieldValue;

static uint64_t m_for(uint64_t n) {
  return 2*((n+1)/4)+1;
}

template <typename FieldT>
struct PrimeContext {
  uint64_t n, m, p, //n = number of veritcies //obvious
  p_dash, r, r2; // montgomery stuff, a _M suffix implies something is in montgomery form
  FieldT *jk_prod_M, // cache of w^j*w^-k / (w^-j*w^k + w^j*w^-k)
  *nat_M, // natural numbers up to n (inclusive)
  *nat_inv_M, // inverses of natural numbers up to n (inclusive)
  *jk_sums_M, // w^-j*w^k + w^j*w^-k 
  *ws_M, // powers of omega (m form)
  *fact_M, // i! for i <= n
  *fact_inv_M; // 1/i! for i <= n

  using TField = FieldT;
};

using prim_ctx_t = PrimeContext<CyclomaticFieldValue>;

// index into rectangular array
static inline size_t jk_pos(size_t j, size_t k, uint64_t m) {
  return j*m + k;
}

// shared over threads - not mutated
static prim_ctx_t *prim_ctx_new(uint64_t n, uint64_t m, uint64_t p, uint64_t w) {
  (void) w;
  prim_ctx_t *ctx = (prim_ctx_t *) malloc(sizeof(prim_ctx_t));
  assert(ctx);
  ctx->n = n;
  ctx->m = m;
  ctx->p = p;
  ctx->p_dash = (uint64_t)(-inv64_u64(p));
  ctx->r = ((uint128_t)1 << 64) % p;
  ctx->r2 = (uint128_t)ctx->r * ctx->r % p;

  // initialize roots of unity
  ctx->ws_M = (FieldT*) malloc(m*sizeof(FieldT));
  assert(ctx->ws_M);
  ctx->ws_M[0] = FieldT::One(m);
  ctx->ws_M[1] = FieldT::Omega(m);
  for (size_t i = 2; i < m; ++i) {
    ctx->ws_M[i] = FieldT::Multiply(ctx->ws_M[i-1], ctx->ws_M[1]);
  }

  // w^j * w^-k lookup - not actually inserted into the context
  FieldT jk_pairs_M[m*m];
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k) {
      jk_pairs_M[jk_pos(j, k, m)] = FieldT::Multiply(ctx->ws_M[j], ctx->ws_M[k ? m-k : 0]);
    }
  }

  // cache of // w^-j*w^k + w^j*w^-k 
  ctx->jk_sums_M = (FieldT*) malloc(m*m*sizeof(FieldT));
  assert(ctx->jk_sums_M);
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k)
      ctx->jk_sums_M[jk_pos(j, k, m)] =
        FieldT::Add(jk_pairs_M[jk_pos(j, k, m)], jk_pairs_M[jk_pos(k, j, m)]);
  }

  // cache of w^j*w^-k / (w^-j*w^k + w^j*w^-k)
  ctx->jk_prod_M = (FieldT*) malloc(m*m*sizeof(FieldT));
  assert(ctx->jk_prod_M);
  for (size_t j = 0; j < m; ++j) {
    for (size_t k = 0; k < m; ++k) {
      size_t pos = jk_pos(j, k, m);
      auto sum_inv = FieldT::Invert(ctx->jk_sums_M[pos]);
      ctx->jk_prod_M[pos] = FieldT::Multiply(jk_pairs_M[pos], sum_inv);
    }
  }

  // 1 to n
  ctx->nat_M = (FieldT*) malloc((n+1)*sizeof(FieldT));
  assert(ctx->nat_M);
  ctx->nat_M[0] = FieldT::One(m);
  for (size_t i = 1; i <= n; ++i) {
    ctx->nat_M[i] = FieldT::Add(FieldT::One(m), ctx->nat_M[i-1]);
  }

  // 1/i for i = 1 to n
  ctx->nat_inv_M = (FieldT*) malloc((n + 1) * sizeof(FieldT));
  assert(ctx->nat_inv_M);
  ctx->nat_inv_M[0] = 0;
  for (size_t k = 1; k <= n; ++k) {
    ctx->nat_inv_M[k] = FieldT::Invert(ctx->nat_M[k]);
  }

  // i! for i = 1 to n+1
  ctx->fact_M = (FieldT*) malloc((n+1)*sizeof(FieldT));
  assert(ctx->fact_M);
  ctx->fact_M[0] = FieldT::One(m);
  for (size_t i = 1; i < n+1; ++i) {
    ctx->fact_M[i] = FieldT::Multiply(ctx->fact_M[i-1], ctx->nat_M[i]);
  }

  // 1/i! for i = 1 to n+1
  ctx->fact_inv_M = (FieldT*) malloc((n+1)*sizeof(FieldT));
  assert(ctx->fact_inv_M);
  ctx->fact_inv_M[n] = FieldT::Invert(ctx->fact_M[n]);
  for (size_t i = n; i; --i) {
    ctx->fact_inv_M[i-1] = FieldT::Multiply(ctx->fact_inv_M[i], ctx->nat_M[i]);
  }

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

// Calculate the multinomal coefficient where the powers of x_i are given by *ms
static FieldT multinomial_coeff(const prim_ctx_t *ctx, const size_t *multiplicities, size_t len) {
  // const uint64_t p = ctx->p, p_dash = ctx->p_dash;

  FieldT coeff = ctx->fact_M[ctx->n - 1];

  // In our new field it'd be faster to calc this as an int.
  for (size_t i = 0; i < len; ++i) {
    coeff.MultiplyBy(ctx->fact_inv_M[multiplicities[i]]);
  }

  return coeff;
}

// Compute the determinant via gaussian elimination
// we cary the denominator through and only compute a single inverse at the end
template <typename FieldT>
static FieldT det_mod_p(FieldT *A, size_t dim, const prim_ctx_t *ctx) {
  const uint64_t p = ctx->p, p_dash = ctx->p_dash;
  auto det = FieldT::One(ctx->m);
  auto scaling_factor = FieldT::One(ctx->m);

  for (size_t k = 0; k < dim; ++k) {
    size_t pivot_i = k;
    // If the cell on the diagonal we're about to pivot off is zero - find the next row with a non zero entry in that col
    while (pivot_i < dim && A[pivot_i*dim + k].IsZero()){
      ++pivot_i;
    }
    // if there was no non-zero cell - det is zero
    if (pivot_i == dim) {
      return 0;
    }

    
    if (pivot_i != k) {
      // We swap the rows over so that we have a non zero el on the diagonal
      for (size_t j = 0; j < dim; ++j) {
        FieldT tmp = A[k*dim + j];
        A[k*dim + j] = A[pivot_i*dim + j];
        A[pivot_i*dim + j] = tmp;
      }
      det = FieldT::Negate(det); // And flip the sign of the determinant
    }

    FieldT pivot = A[k*dim + k];
    // multiply in our value on diagonal
    det.MultiplyBy(pivot);

    // Now do the elimination
    for (size_t i = k + 1; i < dim; ++i) {
      // Rather than do division on each row we multiply each row up to a common factor
      // scaling factor is where we record the product of thse numbers so we can divide though by
      // at the end to compensate
      scaling_factor.MultiplyBy(pivot);
      FieldT multiplier = A[i*dim + k];
      for (size_t j = k; j < dim; ++j)
      // mul and subtract off the rest
        A[i*dim + j] = FieldT::Subtract(FieldT::Multiply((A[i*dim + j], pivot), 
        FieldT::Multiply( A[k*dim + j], multiplier)));
    }
  }

  return det.MultiplyBy(FieldT::Invert(scaling_factor));
}

template <typename FieldT>
static FieldT f_fst_term(uint64_t * multiplicities, const prim_ctx_t *ctx) {
  const uint64_t m = ctx->m, p = ctx->p, p_dash = ctx->p_dash;
  FieldT acc = FieldT::One(m);

  for (size_t a = 0; a < m; ++a) {
    uint64_t ca = multiplicities[a];
    if (ca >= 2) {
      FieldT base = ctx->jk_sums_M[jk_pos(a, a, m)];
      uint64_t e = (ca*(ca-1)) / 2;
      acc.MultiplyBy(field_pow(base, e));
    }
  }

  for (size_t a = 0; a < m; ++a) {
    uint64_t ca = multiplicities[a];
    if (!ca) {
      continue;
    }
    for (size_t b = a+1; b < m; ++b) {
      uint64_t cb = multiplicities[b];
      if (!cb) {
        continue;
      }
      acc.MultiplyBy(field_pow(ctx->jk_sums_M[jk_pos(a, b, m)], ca*cb));
    }
  }

  return acc;
}

static FieldT f_snd_trm(uint64_t *multiplicities, const prim_ctx_t *ctx) {
  const uint64_t m = ctx->m;

  // active groups
  // This is the indexs of the non zero elements of c
  // Which is also the powers of w they correspond to
  // typ for "type" aka class / colour
  size_t typ[m], r = 0;
  for (size_t i = 0; i < m; ++i) {
    if (multiplicities[i]) {
      typ[r] = i;
      ++r;
    }
  }

  auto prod_M = FieldT::One(m);
  
  // for each non zero power of omega w^i in our args
  for (size_t a = 0; a < r; ++a) {
    size_t i = typ[a];

    // This is basically the column sum
    auto sum = FieldT::Zero(m);
    // for each non zero power of omega w^j in our args
    for (size_t b = 0; b < r; ++b) {
      size_t j = typ[b];
      // w^i*w^-j
      // (n.b. we could distribute over w^i here if we wanted)

      // This is the inner element of A matrix from the paper (up to sign) / inner element of product sum in paper
      auto w = ctx->jk_prod_M[jk_pos(i, j, m)];

      // sum += w * multiplicity of (w^j)
      sum.Accumulate(FieldT::Multiply(ctx->nat_M[multiplicities[j]], w));

      // let B_jk = w^j*w^-k / (w^-j*w^k + w^j*w^-k)
      // reorder args to f such they increase in order of power of omega
      // Then we have (x_0, x_1, x_2...)
      // Let B_kl = x_k*x_l^-1 / (.....)
      // then sum(k) = sum over l of B_kl
    }

    prod_M.MultiplyBy(field_pow(sum, multiplicities[i]-1));

    // prod_M - product over each deleted row / col (we're leaving 1 behind) 
    // of the sum over l of B_kl
    // So like for each column we "delete" we multiply by the column sum...
  }

  // #### Up to here prod_M is the same
  // Prod M = product[a = 0->r-1] sum[b = 0->1-r] w^coeff_cnt[a] * w^-coeff_cnt[b]

  // #### And here is the point is diverges
  prod_M.MultiplyBy(ctx->nat_inv_M[multiplicities[0]]);

  // Put this isn't the only difference


  // We divide by the multiplicty of 1??

  // quotient minor
  size_t dim = r - 1;
  // If all terms were the same power of w and we quotiented everything out
  if (!dim)
    return prod_M;

  // We're constructing the minor of the reduced matrix
  // dropping the first row / col
  prim_ctx_t::TField A[dim*dim];
  for (size_t a = 1; a < r; ++a) {
    size_t i = typ[a];

    // contribution from the deleted block?????
    FieldT w_del = ctx->jk_prod_M[jk_pos(i, 0, m)];

    // When making the "laplaican" the diagonal is the sum of all off-diagonal elements in the row of the full matrix
    // including the column dropped to form the minor. So we special add that to the diag here (with multiplicity)
    FieldT diag = FieldT::Multiply(ctx->nat_M[multiplicities[0]], w_del);
    
    // remaining off-diagonal blocks
    for (size_t b = 1; b < r; ++b) {
      size_t j = typ[b];
      if (j == i)
        continue;

      FieldT w = ctx->jk_prod_M[jk_pos(i, j, m)];

      // Again fill the matrix as per it's normal terms but with multiplicity
      FieldT v = FieldT::Multiply(ctx->nat_M[multiplicities[j]], w);

      // This is the -1 coefficient on the off diag elements
      A[(a-1)*dim + (b-1)] = FieldT::Negate(v);
      // and add it to the total for the diag elements
      diag.Accumulate(v);
    }

    A[(a-1)*dim + (a-1)] = diag;
  }

  // I guess prod_M is some magic compensation coefficient
  return FieldT::Multiply(prod_M, det_mod_p(A, dim, ctx));
}

template <typename FieldT>
static FieldT f(uint64_t * multiplicities, const prim_ctx_t *ctx) {
  return FieldT::Multiply(f_fst_term<FieldT>(multiplicities, ctx), f_snd_trm<FieldT>(multiplicities, ctx));
}

// state of whole process (shared over threads)
typedef struct {
  uint64_t n, /* element of seq */ m, /*w is the mth root of unity*/ *ps /* list of primes */;
  size_t idx /* withth prime*/, np /* number of primes*/, *vecss /*Some buffers for vectors*/;
  bool quiet, snapshot; /*mode stuff - saving snapshots so you can restart*/
  size_t n_thrds; /*number of threads*/
} proc_state_t;

template <typename FieldT>
struct Worker{
  Atomic *done;
  const prim_ctx_t *ctx;
  queue_t *q;
  size_t *vecs;
  FieldT *l_acc, *acc;
  pthread_mutex_t *acc_mu;
  bool idle;
};

using worker_t = Worker<CyclomaticFieldValue>;

static void w_resume(void *ud) {
  worker_t *w = (worker_t *)ud;
  w->idle = false;
}

static resume_cb_t w_idle(void *ud) {
  worker_t *w = (worker_t *)ud;
  pthread_mutex_lock(w->acc_mu);
  w->acc->Accumulate(*(w->l_acc));
  *(w->l_acc) = std::remove_reference_t<decltype(*(w->l_acc))>::Zero(w->l_acc->mCoefficients.size());
  pthread_mutex_unlock(w->acc_mu);
  w->idle = true;
  return w_resume;
}

// thread function for doing work on the "maths stuff"
static void *residue_for_prime(void *ud) {
  worker_t *worker = (worker_t *)ud;
  const prim_ctx_t *ctx = worker->ctx;
  uint64_t m = ctx->m; //p = ctx->p, 
  FieldT l_acc = 0; // l_acc is where we're going to accumulate the total residual from the stuff we pull from our work queue
  size_t *vecs = worker->vecs;
  worker->l_acc = &l_acc;

  for (;;) {
    size_t n_vec = queue_pop(worker->q, vecs, w_idle, worker);
    if (!n_vec) break;

    for (size_t c = 0; c < n_vec; ++c) {
      // each vector has len m
      // each vector contains the multiplicity with which each power of omega appears in the args
      // i.e. 1 3 7 = 1 lot of w^0, 3 lots of w^1, 7 lots of w^2
      size_t *vec = &vecs[c*m];
      auto f_0 = f<FieldT>(vec, ctx);
      auto const coeff_baseline = multinomial_coeff(ctx, vec, m);

      // Loop over each "rotation" of the vector of argument multiplicities. This is
      // equivilent to multiplying all the coefficients by w
      for (size_t r = 0; r < m; ++r) {
        // We require there always be at least one "1" in the arguments to f() (per the paper)
        // that is to say if the multiplicty of "1" arguments is zero - we should skip this case
        if (vec[r] == 0) continue;

        // The multinomial coefficient would be constant over all "rotations" of the multiplicities
        // but because we're assuming at least one argument is always "1" which requires us to subtract
        // 1 from the first multiplicity. Rather than recompute the full coeff each time we can take a
        // baseline "coefficient" and multiply it by j to convert 1/j! to 1/(j-1!)
        auto coeff = FieldT::Multiply(coeff_baseline, ctx->nat_M[vec[r]]);

        size_t idx = (2*r) % m;
        // f_0 = coeff * f_0 * w^(m-idx) = coeff * f_0 * w^-2r
        // This result comes from having to permute one of the ones from one of the first n-1 args into the nth arg
        // See the paper for more info
        auto f_n = FieldT::Multiply(coeff, FieldT::Multiply(f_0, ctx->ws_M[idx ? m-idx : 0]));
        l_acc.Accumulate(f_n);
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

template <typename FieldT>
static int ret(proc_state_t *st, prim_ctx_t *ctx, FieldT const & acc, FieldT *res, FieldT *p_ret) {
  auto denom = FieldT::Invert(field_pow(ctx->nat_M[ctx->m], ctx->n-1));
  auto ret = FieldT::Multiply(acc, denom);
  prim_ctx_free(ctx);

  *p_ret = st->ps[st->idx++];
  *res = ret;
  return 1;
}


// Implementation of the virtual function - entrypoint for calculation
static int proc_next(source_t *self, FieldT *res, uint64_t *p_ret) {
  proc_state_t *st = (proc_state_t *) self->state;
  if (st->idx == st->np) return 0;

  // n, m, prime
  uint64_t n = st->n, m = st->m, p = st->ps[st->idx];
  // root of unity
  uint64_t w = mth_root_mod_p(p, m);
  prim_ctx_t *ctx = prim_ctx_new(n, m, p, w);

  const size_t siz = canon_iter_size(m, n);

  Atomic done = 0;
  FieldT acc = 0;

  uint64_t iter_st[m+5];
  size_t st_len = 0;

  if (st->snapshot) {
    ASSERT(false);
    //snapshot_try_resume(n, p, &done, &acc, iter_st, &st_len);
  }
    
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
    w_ctxs[i] = (worker_t){ .done = &done, .ctx = ctx, .q = q, .vecs = &st->vecss[i*CHUNK*m], .acc = &acc, .acc_mu = &acc_mu, .idle = false };
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
  proc_state_t *st = (proc_state_t *)self->state;
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

  proc_state_t *st = (proc_state_t *)malloc(sizeof(*st));
  assert(st);
  size_t n_thrds = get_num_threads();
  size_t *vecss = (size_t *)malloc(CHUNK*m*(n_thrds+1+Q_CAP)*sizeof(size_t));
  assert(vecss);

  *st = (proc_state_t){ .n = n, .m = m, .ps = ps, .idx = 0, .np = np, .vecss = vecss, .quiet = quiet, .snapshot = snapshot, .n_thrds = n_thrds };

  source_t *src = (source_t *)malloc(sizeof *src);
  assert(src);
  *src = (source_t){ .next = proc_next, .destroy = proc_destroy, .state = st };
  return src;
}
