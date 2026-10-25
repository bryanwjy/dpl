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
    -r linear 0.5:2 -r linear 0.7071:1.4143 -r linear 0.999:1.001 \
    -r urand 0.9375:1.0625 -r linear 1:65536 -r linear 0:1 -r urand 0x1p-126: \
    -r linear -16:-0.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
    -i 1 -i -1 -i -2 -i 2 -i 0.5 -i 1024 -i 0x1p-20 -i 3 -i 10 \
    -i 2.718281828459045 -i 0.36787944117144233 -i 7.38905609893065 \
    -i 1.4142135623730951 -i 0.7071067811865476 -i 0x1p127 -i 0x1.fffffep+127 \
    -i 0x1p-126 -i 0x1p-149 -i 0x1.fffffcp-127 -i 0x1.000002p+0 -i 0x1.fffffep-1 \
    -o $@

$(CURRENT_SOURCE_DIR)/f64.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear 0.5:2 -r linear 0.7071:1.4143 -r linear 0.999:1.001 \
    -r urand 0.9375:1.0625 -r linear 1:65536 -r linear 0:1 -r urand 0x1p-1022: \
    -r linear -16:-0.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
    -i 1 -i -1 -i -2 -i 2 -i 0.5 -i 1024 -i 0x1p-20 -i 3 -i 10 \
    -i 2.718281828459045 -i 0.36787944117144233 -i 7.38905609893065 \
    -i 1.4142135623730951 -i 0.7071067811865476 -i 0x1p1023 \
    -i 0x1.fffffffffffffp+1023 -i 0x1p-1022 -i 0x1p-1074 \
    -i 0x1.ffffffffffffep-1023 -i 0x1.0000000000001p+0 -i 0x1.fffffffffffffp-1 \
    -o $@

$(CURRENT_SOURCE_DIR)/f16.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear 0.5:2 -r linear 0.7071:1.4143 -r linear 0.99:1.01 \
    -r urand 0.9375:1.0625 -r linear 1:1024 -r linear 0:1 -r urand 0x1p-14: \
    -r linear -16:-0.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
    -i 1 -i -1 -i -2 -i 2 -i 0.5 -i 1024 -i 0x1p-20 -i 3 -i 10 \
    -i 2.718281828459045 -i 0.36787944117144233 -i 7.38905609893065 \
    -i 1.4142135623730951 -i 0.7071067811865476 -i 0x1p15 -i 65504 -i 0x1p-14 \
    -i 0x1p-24 -i 0x1.ff8p-15 -i 0x1.004p+0 -i 0x1.ffcp-1 \
    -o $@

$(CURRENT_SOURCE_DIR)/bf16.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear 0.5:2 -r linear 0.7071:1.4143 -r linear 0.97:1.03 \
    -r urand 0.9375:1.0625 -r linear 1:65536 -r linear 0:1 -r urand 0x1p-126: \
    -r linear -16:-0.001 -r urand : -i 0 -i -0 -i infinity -i -infinity -i nan \
    -i 1 -i -1 -i -2 -i 2 -i 0.5 -i 1024 -i 0x1p-20 -i 3 -i 10 \
    -i 2.718281828459045 -i 0.36787944117144233 -i 7.38905609893065 \
    -i 1.4142135623730951 -i 0.7071067811865476 -i 0x1p127 -i 0x1.fep+127 \
    -i 0x1p-126 -i 0x1p-133 -i 0x1.fcp-127 -i 0x1.02p+0 -i 0x1.fep-1 \
    -o $@
