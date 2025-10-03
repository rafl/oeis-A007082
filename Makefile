DEBUG := 1
SLOW_DIVISION := 0

-include .config

CFLAGS := -g -std=gnu18 -O0 -march=native -flto -Wall -Wextra \
          $(shell pkg-config gmp --cflags) -Iinclude -MMD -MP \
          -DDEBUG=$(DEBUG) -DSLOW_DIVISION=$(SLOW_DIVISION) -m64
LDFLAGS := $(shell pkg-config gmp --libs) -lm

PGO ?= none
PGO_DIR ?= build/pgo
PROFDATA := $(PGO_DIR)/default.profdata
IS_CLANG := $(shell $(CC) --version 2>/dev/null | grep -q clang && echo 1)

ifeq ($(PGO),gen)
CFLAGS += -fprofile-generate=$(PGO_DIR)
LDFLAGS += -fprofile-generate=$(PGO_DIR)
endif

ifeq ($(PGO),use)
CFLAGS += -fprofile-use=$(PGO_DIR) -fprofile-correction
LDFLAGS += -fprofile-use=$(PGO_DIR)
endif

SRC_DIR := src
OBJ_DIR := build/obj

SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

MAIN_FILES := oeis.c nnecklaces.c
MAIN_OBJS := $(patsubst %.c,$(OBJ_DIR)/%.o,$(MAIN_FILES))

UTIL_OBJS := $(filter-out $(MAIN_OBJS),$(OBJS))

TARGETS := $(patsubst %.c,%,$(MAIN_FILES))

.PHONY: all gen use optimised clean pgo-clean test fmt

all: $(if $(filter use,$(PGO)),$(PROFDATA)) $(TARGETS)

$(PROFDATA):
	[ "$(IS_CLANG)" != "1" ] || llvm-profdata merge -output=$@ $(PGO_DIR)/default_*.profraw

gen:
	$(MAKE) clean
	$(MAKE) PGO=gen oeis

use:
	$(MAKE) clean
	$(MAKE) PGO=use oeis

optimised:
	$(MAKE) gen
	./oeis 19
	./oeis --jack offset 19
	$(MAKE) use

$(TARGETS): %: $(OBJ_DIR)/%.o $(UTIL_OBJS) | $(PGO_DIR)
	$(CC) $(CFLAGS) $< $(UTIL_OBJS) $(LDFLAGS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR) $(PGO_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR) $(PGO_DIR):
	@mkdir -p $@

-include $(DEPS)

clean:
	@rm -rf $(OBJ_DIR) $(TARGETS)

pgo-clean:
	@rm -rf $(PGO_DIR)

test: oeis
	@bash test.sh $(N)

fmt:
	bash -c 'shopt -s globstar; clang-format -i -- src/**.c include/**.h'
