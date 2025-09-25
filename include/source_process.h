#pragma once

#include "source.h"
#include <stdbool.h>
#include <stddef.h>

typedef enum {
  PROC_MODE_REG = 0,
  PROC_MODE_JACK_OFFSET = (1 << 0),
  PROC_MODE_JACKEST = (1 << 1),
  PROC_MODE_JACKBOTH = PROC_MODE_JACK_OFFSET|PROC_MODE_JACKEST,
} process_mode_t;

source_t *source_process_new(process_mode_t, uint64_t, uint64_t, bool, bool, size_t *);
size_t *source_process_vecss(source_t *);
