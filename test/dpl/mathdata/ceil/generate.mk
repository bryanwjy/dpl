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
	@$(MPFR_BIN) $(CURRENT_FUNCTION) f32 -n $(COUNT) --seed $(SEED) \
    -r urand -0x1p23:-1 -r urand 1:0x1p23 \
    -r urand -1:1 -r urand 0x1p22:infinity -r urand -infinity:-0x1p22 \
    -o $@

$(CURRENT_SOURCE_DIR)/f64.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r urand -0x1p52:-1 -r urand 1:0x1p52 -r urand -1:1 \
    -r urand 0x1p51:infinity -r urand -infinity:-0x1p51 \
    -o $@

$(CURRENT_SOURCE_DIR)/f16.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r urand -0x1p10:-1 -r urand 1:0x1p10 -r urand -1:1 \
    -r urand 0x1p9:infinity -r urand -infinity:-0x1p9 \
    -o $@

$(CURRENT_SOURCE_DIR)/bf16.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r urand -0x1p7:-1 -r urand 1:0x1p7 -r urand -1:1 \
    -r urand 0x1p6:infinity -r urand -infinity:-0x1p6 \
    -o $@
