# Copyright 2026 Bryan Wong

CURRENT_LIST_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_SOURCE_DIR := $(patsubst %/,%,$(dir $(CURRENT_LIST_PATH)))
CURRENT_FUNCTION := $(notdir $(CURRENT_SOURCE_DIR))

$(CURRENT_SOURCE_DIR)/%: private CURRENT_FUNCTION := $(CURRENT_FUNCTION)
SEED := $$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
COUNT := 12
TYPES := f32 f64 f16 bf16
CURRENT_TARGETS := $(addprefix $(CURRENT_SOURCE_DIR)/,$(addsuffix .h, $(TYPES)))

all: $(CURRENT_TARGETS)

.PHONY: $(CURRENT_TARGETS)


$(CURRENT_SOURCE_DIR)/f32.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear 0.5:2 -4:4 -r linear 0.999:1.001 -10000:10000 \
    -r urand 0x1p-8:0x1p8 -8:8 -r linear -4:-0.25 -4:4 -r urand 0x1p-126:1 0:1 \
    -r linear 1.99:2 120:128 -r linear 2:2.01 128:256 \
    -r linear 2:2.01 -149:-126 -r linear 2:2.01 -300:-150 -r urand : : -i nan 0 \
    -i infinity -0 -i 1 nan -i 1 infinity -i 1 -infinity -i -1 infinity \
    -i -1 -infinity -i 0 -1 -i -0 -1 -i -0 -2 -i 0 -0.5 -i -0 -infinity -i 0 3 \
    -i -0 3 -i -0 2 -i -0 0.5 -i -2 3 -i -2 -3 -i -2 2 -i -2 0.5 -i -1.5 31 \
    -i -1 0x1p14 -i 0.5 infinity -i 0.5 -infinity -i 2 infinity -i 2 -infinity \
    -i -infinity 3 -i -infinity 2 -i -infinity -3 -i -infinity -2 \
    -i infinity -1 -i infinity 0.5 -i nan 1 -i 2 nan -i 2 10 -i 4 0.5 -i 9 0.5 \
    -i 10 2 -i 3 -2 \
    -o $@

$(CURRENT_SOURCE_DIR)/f64.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear 0.5:2 -4:4 -r linear 0.999:1.001 -100000:100000 \
    -r urand 0x1p-8:0x1p8 -8:8 -r linear -4:-0.25 -4:4 -r urand 0x1p-1022:1 0:1 \
    -r linear 1.99:2 1016:1024 -r linear 2:2.01 1024:2048 \
    -r linear 2:2.01 -1074:-1022 -r linear 2:2.01 -2150:-1075 -r urand : : \
    -i nan 0 -i infinity -0 -i 1 nan -i 1 infinity -i 1 -infinity \
    -i -1 infinity -i -1 -infinity -i 0 -1 -i -0 -1 -i -0 -2 -i 0 -0.5 \
    -i -0 -infinity -i 0 3 -i -0 3 -i -0 2 -i -0 0.5 -i -2 3 -i -2 -3 -i -2 2 \
    -i -2 0.5 -i -1.5 31 -i -1 0x1p14 -i 0.5 infinity -i 0.5 -infinity \
    -i 2 infinity -i 2 -infinity -i -infinity 3 -i -infinity 2 -i -infinity -3 \
    -i -infinity -2 -i infinity -1 -i infinity 0.5 -i nan 1 -i 2 nan -i 2 10 \
    -i 4 0.5 -i 9 0.5 -i 10 2 -i 3 -2 \
    -o $@

$(CURRENT_SOURCE_DIR)/f16.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear 0.5:2 -4:4 -r linear 0.99:1.01 -500:500 \
    -r urand 0x1p-4:0x1p4 -3:3 -r linear -4:-0.25 -4:4 -r urand 0x1p-14:1 0:1 \
    -r linear 1.99:2 8:16 -r linear 2:2.01 16:32 -r linear 2:2.01 -24:-14 \
    -r linear 2:2.01 -50:-25 -r urand : : -i nan 0 -i infinity -0 -i 1 nan \
    -i 1 infinity -i 1 -infinity -i -1 infinity -i -1 -infinity -i 0 -1 \
    -i -0 -1 -i -0 -2 -i 0 -0.5 -i -0 -infinity -i 0 3 -i -0 3 -i -0 2 \
    -i -0 0.5 -i -2 3 -i -2 -3 -i -2 2 -i -2 0.5 -i -1.5 31 -i -1 0x1p14 \
    -i 0.5 infinity -i 0.5 -infinity -i 2 infinity -i 2 -infinity \
    -i -infinity 3 -i -infinity 2 -i -infinity -3 -i -infinity -2 \
    -i infinity -1 -i infinity 0.5 -i nan 1 -i 2 nan -i 2 10 -i 4 0.5 -i 9 0.5 \
    -i 10 2 -i 3 -2 \
    -o $@

$(CURRENT_SOURCE_DIR)/bf16.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear 0.5:2 -4:4 -r linear 0.97:1.03 -200:200 \
    -r urand 0x1p-8:0x1p8 -8:8 -r linear -4:-0.25 -4:4 -r urand 0x1p-126:1 0:1 \
    -r linear 1.99:2 120:128 -r linear 2:2.01 128:256 \
    -r linear 2:2.01 -133:-126 -r linear 2:2.01 -268:-134 -r urand : : -i nan 0 \
    -i infinity -0 -i 1 nan -i 1 infinity -i 1 -infinity -i -1 infinity \
    -i -1 -infinity -i 0 -1 -i -0 -1 -i -0 -2 -i 0 -0.5 -i -0 -infinity -i 0 3 \
    -i -0 3 -i -0 2 -i -0 0.5 -i -2 3 -i -2 -3 -i -2 2 -i -2 0.5 -i -1.5 31 \
    -i -1 0x1p14 -i 0.5 infinity -i 0.5 -infinity -i 2 infinity -i 2 -infinity \
    -i -infinity 3 -i -infinity 2 -i -infinity -3 -i -infinity -2 \
    -i infinity -1 -i infinity 0.5 -i nan 1 -i 2 nan -i 2 10 -i 4 0.5 -i 9 0.5 \
    -i 10 2 -i 3 -2 \
    -o $@
