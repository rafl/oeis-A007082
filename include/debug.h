#pragma once

#ifndef DEBUG
#define DEBUG 1
#endif

#if DEBUG
#define VERIFY(e)                                                              \
  do {                                                                         \
    bool _verify_ok = !!(e);                                                   \
    assert(_verify_ok);                                                        \
  } while (0);
#define DEBUG_ARG
#else
#define NDEBUG
#define VERIFY(e) ((void)(e))
#define DEBUG_ARG __attribute__((unused))
#endif

#include <assert.h>
