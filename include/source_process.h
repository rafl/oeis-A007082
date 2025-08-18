#pragma once

#include "source.h"
#include <stdbool.h>

source_t *source_process_new(uint64_t, uint64_t, bool, bool);

void jack_test();