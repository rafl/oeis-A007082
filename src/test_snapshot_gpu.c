// Snapshot stress test for GPU workers
// Tests that resuming from snapshots produces correct results
//
// Approach:
// 1. Run computation once to completion with SNAPSHOT_MULTI=1
//    This creates numbered snapshots .ss.0, .ss.1, .ss.2, etc.
// 2. After completion, iterate through all snapshot files
// 3. Resume from each and verify result matches expected

#include "maths.h"
#include "primes.h"
#include "source.h"
#include "source_process.h"

#ifdef USE_GPU
#include "gpu_det.h"
#endif

#include <dirent.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>
#include <time.h>
#include <errno.h>

// Global for cleanup
static char g_temp_dir[256] = {0};

// Test configuration
#define TEST_N 25        // n value (small but exercises GPU)
#define TEST_M_ID 0      // m_id (prime stride index)

// Run computation and return result for first prime
// Returns result in *out, returns 1 on success, 0 on failure
static int run_computation(uint64_t n, uint64_t m_id, bool snapshot,
                           uint64_t *out, uint64_t *prime_out) {
  source_t *src = source_process_new(PROC_MODE_REG, n, m_id, true, snapshot, NULL);
  if (!src) {
    fprintf(stderr, "Failed to create source\n");
    return 0;
  }

  uint64_t result, prime;
  int ok = src->next(src, &result, &prime);
  src->destroy(src);

  if (ok) {
    *out = result;
    if (prime_out) *prime_out = prime;
    return 1;
  }
  return 0;
}


// Build snapshot path prefix for globbing
static void get_snapshot_prefix(uint64_t n, uint64_t p, char *buf, size_t len) {
  snprintf(buf, len, ".%" PRIu64 ".%" PRIu64 ".ss.", n, p);
}

// Compare function for qsort
static int int_cmp(const void *a, const void *b) {
  return *(const int *)a - *(const int *)b;
}

// Count snapshot files and return array of their numbers (sorted)
// Caller must free the returned array
static int *find_snapshot_numbers(uint64_t n, uint64_t p, int *count) {
  char prefix[256];
  get_snapshot_prefix(n, p, prefix, sizeof(prefix));
  size_t prefix_len = strlen(prefix);

  // First pass: count files
  *count = 0;
  DIR *dir = opendir(".");
  if (!dir) return NULL;

  struct dirent *ent;
  while ((ent = readdir(dir)) != NULL) {
    if (strncmp(ent->d_name, prefix, prefix_len) == 0) {
      (*count)++;
    }
  }

  if (*count == 0) {
    closedir(dir);
    return NULL;
  }

  // Allocate and second pass: collect numbers
  int *numbers = malloc(*count * sizeof(int));
  rewinddir(dir);
  int idx = 0;
  while ((ent = readdir(dir)) != NULL) {
    if (strncmp(ent->d_name, prefix, prefix_len) == 0) {
      numbers[idx++] = atoi(ent->d_name + prefix_len);
    }
  }
  closedir(dir);

  // Sort for predictable order
  qsort(numbers, *count, sizeof(int), int_cmp);

  return numbers;
}

// Remove temp directory and all contents
static void cleanup_temp_dir(void) {
  if (g_temp_dir[0] == '\0') return;

  DIR *dir = opendir(g_temp_dir);
  if (!dir) return;

  struct dirent *ent;
  char path[512];
  while ((ent = readdir(dir)) != NULL) {
    if (strcmp(ent->d_name, ".") == 0 || strcmp(ent->d_name, "..") == 0)
      continue;
    snprintf(path, sizeof(path), "%s/%s", g_temp_dir, ent->d_name);
    unlink(path);
  }
  closedir(dir);
  rmdir(g_temp_dir);
  g_temp_dir[0] = '\0';
}

// Create temp directory and chdir into it
static int setup_temp_dir(void) {
  snprintf(g_temp_dir, sizeof(g_temp_dir), "/tmp/snapshot_test_%d_%ld",
           (int)getpid(), (long)time(NULL));

  if (mkdir(g_temp_dir, 0755) != 0) {
    fprintf(stderr, "Failed to create temp dir %s: %s\n", g_temp_dir, strerror(errno));
    g_temp_dir[0] = '\0';
    return 0;
  }

  if (chdir(g_temp_dir) != 0) {
    fprintf(stderr, "Failed to chdir to %s: %s\n", g_temp_dir, strerror(errno));
    rmdir(g_temp_dir);
    g_temp_dir[0] = '\0';
    return 0;
  }

  return 1;
}

// Copy a numbered snapshot to the base snapshot file for resume
static int setup_resume_snapshot(uint64_t n, uint64_t p, int num) {
  char src_path[256], dst_path[256];
  snprintf(src_path, sizeof(src_path), ".%" PRIu64 ".%" PRIu64 ".ss.%d", n, p, num);
  snprintf(dst_path, sizeof(dst_path), ".%" PRIu64 ".%" PRIu64 ".ss", n, p);

  // Copy file
  FILE *src = fopen(src_path, "rb");
  if (!src) return 0;

  FILE *dst = fopen(dst_path, "wb");
  if (!dst) {
    fclose(src);
    return 0;
  }

  char buf[4096];
  size_t n_read;
  while ((n_read = fread(buf, 1, sizeof(buf), src)) > 0) {
    if (fwrite(buf, 1, n_read, dst) != n_read) {
      fclose(src);
      fclose(dst);
      return 0;
    }
  }

  fclose(src);
  fclose(dst);
  return 1;
}

int main(int argc, char **argv) {
  (void)argc;
  (void)argv;

  uint64_t n = TEST_N;
  uint64_t m_id = TEST_M_ID;

  printf("Snapshot stress test\n");
  printf("Parameters: n=%" PRIu64 ", m_id=%" PRIu64 "\n", n, m_id);

  // Set up temp directory
  if (!setup_temp_dir()) {
    return 1;
  }
  printf("Working in: %s\n", g_temp_dir);

  // Set environment for aggressive snapshotting
  setenv("SNAPSHOT_INTERVAL_MS", "10", 1);
  setenv("SNAPSHOT_MULTI", "1", 1);

  // Run computation once to completion with snapshots enabled
  // This will create multiple numbered snapshots
  printf("Running initial computation with snapshots...\n");
  uint64_t expected, p;
  if (!run_computation(n, m_id, true, &expected, &p)) {
    printf("ERROR: Failed to compute expected result\n");
    cleanup_temp_dir();
    return 1;
  }
  printf("Prime: %" PRIu64 ", Expected result: %" PRIu64 "\n", p, expected);

  // Find all snapshot files that were created
  int n_snapshots;
  int *snapshot_nums = find_snapshot_numbers(n, p, &n_snapshots);

  if (n_snapshots == 0) {
    printf("WARNING: No snapshots were captured (computation too fast?)\n");
    printf("Try increasing TEST_N or decreasing SNAPSHOT_INTERVAL_MS\n");
    cleanup_temp_dir();
    return 1;
  }

  printf("Found %d snapshots to test\n", n_snapshots);

  // Disable multi-snapshot for resume tests
  unsetenv("SNAPSHOT_MULTI");

  // Test each snapshot
  for (int i = 0; i < n_snapshots; i++) {
    int num = snapshot_nums[i];

    // Copy numbered snapshot to base file for resume
    if (!setup_resume_snapshot(n, p, num)) {
      printf("FAIL: Could not setup snapshot %d for resume\n", num);
      free(snapshot_nums);
      cleanup_temp_dir();
      return 1;
    }

    // Resume and verify
    uint64_t resumed;
    if (!run_computation(n, m_id, true, &resumed, NULL)) {
      printf("FAIL snapshot %d: resume failed\n", num);
      free(snapshot_nums);
      cleanup_temp_dir();
      return 1;
    }

    if (resumed != expected) {
      printf("FAIL snapshot %d: got %" PRIu64 ", expected %" PRIu64 "\n",
             num, resumed, expected);
      free(snapshot_nums);
      cleanup_temp_dir();
      return 1;
    }

    if ((i + 1) % 10 == 0 || i + 1 == n_snapshots) {
      printf("  Tested %d/%d snapshots\r", i + 1, n_snapshots);
      fflush(stdout);
    }
  }

  printf("\n");
  free(snapshot_nums);

  // Clean up
  cleanup_temp_dir();

  printf("PASS: All %d snapshots resumed correctly\n", n_snapshots);
  return 0;
}
