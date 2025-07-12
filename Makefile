CFLAGS := -g -std=gnu18 -O0 -march=native -flto -fopenmp -Wall -Wextra -fanalyzer -fstack-protector-strong -D_FORTIFY_SOURCE=2\
          $(shell pkg-config gmp --cflags) -Iinclude -MMD -MP
LDFLAGS := $(shell pkg-config gmp --libs) -lm

PGO ?= none
PGO_DIR ?= build/pgo

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

TARGET := oeis

.PHONY: all gen use optimised clean pgo-clean test

all: $(TARGET)

gen:
	$(MAKE) clean
	$(MAKE) PGO=gen all

use:
	$(MAKE) clean
	$(MAKE) PGO=use all

optimised:
	$(MAKE) gen
	./oeis 17
	$(MAKE) use

$(TARGET): $(OBJS) | $(PGO_DIR)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR) $(PGO_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR) $(PGO_DIR):
	@mkdir -p $@

-include $(DEPS)

clean:
	@rm -rf $(OBJ_DIR) $(TARGET)

pgo-clean:
	@rm -rf $(PGO_DIR)

test: $(TARGET)
	@bash test.sh
