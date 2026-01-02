#pragma once

#include "source.h"
#include "source_process.h"
#include <stdbool.h>

source_t *source_gpu_new(process_mode_t, uint64_t, uint64_t, bool);
