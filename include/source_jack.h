#pragma once

#include "source.h"
#include "source_process.h"
#include <stdbool.h>

source_t *source_jack_new(process_mode_t, uint64_t, uint64_t, bool, bool,
                          mss_el_t *);
