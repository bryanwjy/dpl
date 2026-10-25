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
    -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 -r linear -0.2929:0.4143 \
    -r linear -1:-0.5 -r linear -1:-0.99 -r linear 0.5:16 -r urand 1: \
    -r linear -16:-1.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
    -i -1 -i -2 -i 1 -i -0.5 -i 1.718281828459045 -i 0.41421356237309515 \
    -i -0.2928932188134524 -i 0x1p-30 -i -0x1p-30 -i 0x1p-126 -i 0x1p-149 \
    -i -0x1p-149 -i 0x1.fffffcp-127 -i 0x1.fffffep+127 -i -0x1.fffffep-1 \
    -o $@

$(CURRENT_SOURCE_DIR)/f64.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 -r linear -0.2929:0.4143 \
    -r linear -1:-0.5 -r linear -1:-0.99 -r linear 0.5:16 -r urand 1: \
    -r linear -16:-1.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
    -i -1 -i -2 -i 1 -i -0.5 -i 1.718281828459045 -i 0.41421356237309515 \
    -i -0.2928932188134524 -i 0x1p-30 -i -0x1p-30 -i 0x1p-1022 -i 0x1p-1074 \
    -i -0x1p-1074 -i 0x1.ffffffffffffep-1023 -i 0x1.fffffffffffffp+1023 \
    -i -0x1.fffffffffffffp-1 \
    -o $@

$(CURRENT_SOURCE_DIR)/f16.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 -r linear -0.2929:0.4143 \
    -r linear -1:-0.5 -r linear -1:-0.99 -r linear 0.5:16 -r urand 1: \
    -r linear -16:-1.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
    -i -1 -i -2 -i 1 -i -0.5 -i 1.718281828459045 -i 0.41421356237309515 \
    -i -0.2928932188134524 -i 0x1p-30 -i -0x1p-30 -i 0x1p-14 -i 0x1p-24 \
    -i -0x1p-24 -i 0x1.ff8p-15 -i 65504 -i -0x1.ffcp-1 \
    -o $@

$(CURRENT_SOURCE_DIR)/bf16.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 -r linear -0.2929:0.4143 \
    -r linear -1:-0.5 -r linear -1:-0.99 -r linear 0.5:16 -r urand 1: \
    -r linear -16:-1.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
    -i -1 -i -2 -i 1 -i -0.5 -i 1.718281828459045 -i 0.41421356237309515 \
    -i -0.2928932188134524 -i 0x1p-30 -i -0x1p-30 -i 0x1p-126 -i 0x1p-133 \
    -i -0x1p-133 -i 0x1.fcp-127 -i 0x1.fep+127 -i -0x1.fep-1 \
    -o $@
