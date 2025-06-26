CFLAGS := -g -std=gnu18 -O3 -march=native -flto -fopenmp -Wall -Wextra \
          $(shell pkg-config gmp --cflags) -Iinclude -MMD -MP
LDFLAGS := $(shell pkg-config gmp --libs) -lm

SRC_DIR := src
OBJ_DIR := build
SRCS := $(wildcard $(SRC_DIR)/*.c)
OBJS := $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

TARGET := oeis

.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ_DIR):
	@mkdir -p $@

-include $(DEPS)

.PHONY: clean
clean:
	@rm -rf $(OBJ_DIR) $(TARGET)
