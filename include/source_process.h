#pragma once

#include "source.h"
#include <stdbool.h>

typedef enum {
  PROC_MODE_REG = 0,
  PROC_MODE_JACKOFF = (1 << 0),
  PROC_MODE_JACKEST = (1 << 1),
//  MODE_JACKBOTH = MODE_JACKOFF|MODE_JACKEST,
} process_mode_t;

source_t *source_process_new(process_mode_t, uint64_t, uint64_t, bool, bool);
