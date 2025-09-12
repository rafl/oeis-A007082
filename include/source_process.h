#pragma once

#include "source.h"
#include <stdbool.h>

typedef enum {
  MODE_REG = 0,
  MODE_JACKOFF = (1 << 0),
  MODE_JACKEST = (1 << 1),
//  MODE_JACKBOTH = MODE_JACKOFF|MODE_JACKEST,
} process_mode_t;

source_t *source_process_new(process_mode_t, uint64_t, uint64_t, bool, bool);
