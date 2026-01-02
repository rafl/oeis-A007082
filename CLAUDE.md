# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This project computes values for OEIS sequence A007082 using a multi-threaded C implementation. It calculates residues modulo primes and combines them using the Chinese Remainder Theorem (CRT) to determine exact values.

## Build System

The project uses a Makefile with several important targets:

**Basic commands:**
- `make` - Build all targets (creates `oeis` and `nnecklaces` binaries)
- `make clean` - Remove build artifacts and binaries
- `make test` - Run test suite (optionally specify `N=<num>` to test up to n=num)
- `make fmt` - Format all C source files using clang-format

**Profile-Guided Optimization (PGO):**
- `make optimised` - Build optimized binary using PGO (runs training workload automatically)
- `make gen` - Build for profile generation
- `make use` - Build using collected profiles
- `make pgo-clean` - Remove PGO profile data

**Build configuration:**
- `.config` file (optional) can override `DEBUG` and `SLOW_DIVISION` flags
- Default: `DEBUG=1`, `SLOW_DIVISION=0`
- Dependencies: GMP library (`pkg-config gmp`), math library

## Running the Code

**Main binary:** `oeis`

**Basic usage:**
```bash
./oeis <k>              # Compute e(n) where k = 2n+1
./oeis -q <k>           # Quiet mode - only output final result
./oeis -p <k>           # Process mode only (compute residues)
./oeis -c               # Combine mode only (read residues from stdin)
./oeis -s <k>           # Enable snapshotting for resumable computation
./oeis --jack <mode> <k>  # Use jackknife estimation
```

**Jack modes:**
- `offset` - Jackknife offset estimation
- `est` - Jackknife estimation only
- `both` - Both offset and estimation

**Examples:**
```bash
./oeis 19               # Compute n=9 (k=19)
./oeis -q 19            # Same, quiet output
./oeis --jack both 19   # Use jackknife method
```

## Architecture

### Source Abstraction

The codebase uses a polymorphic "source" pattern (see `include/source.h`) where different computation strategies implement the same interface:
- `source_t` - Base type with `next()` and `destroy()` function pointers
- `source_process_new()` - Creates source that computes residues modulo primes
- `source_jack_new()` - Creates source using jackknife estimation
- `source_stdin_new()` - Creates source that reads residues from stdin

### Two-Phase Pipeline

**Phase 1: Process** (`src/source_process.c`, `src/source_jack.c`)
- Computes residues modulo successive primes
- Multi-threaded using work queue (`src/queue.c`)
- Uses Montgomery arithmetic for modular operations (`include/maths.h`)
- Caches powers of primitive roots and factorial values
- Supports resumable computation via snapshots (`src/snapshot.c`)

**Phase 2: Combine** (`src/combine.c`)
- Uses Chinese Remainder Theorem to combine residues
- Uses GMP for arbitrary-precision arithmetic
- Detects convergence when result stabilizes

### Key Data Structures

- `prim_ctx_t` (in `src/source_process.c`) - Per-prime computation context with Montgomery form caches
- `canon_iter_t` (`include/mss.h`) - Iterator over canonical coefficient sets
- `comb_ctx_t` (`include/combine.h`) - CRT accumulator with GMP integers
- `queue_t` (`include/queue.h`) - Thread-safe work queue
- `snapshot_t` (`include/snapshot.h`) - Periodic checkpointing for long computations
- `progress_t` (`include/progress.h`) - Progress monitoring thread

### Computation Strategy

The algorithm:
1. For each prime p where m = 2⌊(n+1)/4⌋+1 divides p-1:
   - Find primitive m-th root of unity ω
   - Enumerate canonical coefficient sets using MSS iterator
   - Compute polynomial evaluations modulo p using cached Montgomery values
   - Accumulate result modulo p
2. Feed residues to CRT combiner until result converges

### Threading Model

- Main thread dispatches work items to queue
- Worker threads (CPU count) process items from queue
- Progress thread monitors and displays computation status
- Snapshot thread periodically saves state (when `-s` flag used)
- Thread synchronization via pthreads mutexes/conditions

## Testing

- `test.sh` - Test harness comparing computed values against `b007082.txt`
- Runs multiple test cases including both regular and jackknife modes
- Test parameters: `./test.sh [MAX_N]` (defaults to n=7)
- CI runs tests up to n=11 with both `SLOW_DIVISION=0` and `SLOW_DIVISION=1`

## Code Style

- Format: LLVM style via clang-format
- Language: C (gnu18 standard)
- Naming: snake_case for functions/variables, `_t` suffix for types
- Montgomery form values suffixed with `_M`
- Optimization: `-O3 -march=native -flto` enabled by default
