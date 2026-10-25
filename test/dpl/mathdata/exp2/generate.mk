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
    -r linear -1:1 -r urand -0x1p-8:0x1p-8 -r linear 1:32 \
    -r linear -32:-1 -r linear -126:128 -r linear 127:128 \
    -r linear 128:256 -r linear -150:-126 -r linear -300:-150 \
    -r urand : \
    -o $@

$(CURRENT_SOURCE_DIR)/f64.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear -1:1 -r urand -0x1p-8:0x1p-8 -r linear 1:32 \
    -r linear -32:-1 -r linear -1022:1024 -r linear 1023:1024 \
    -r linear 1024:2048 -r linear -1075:-1022 -r linear -2150:-1075 \
    -r urand : \
    -o $@

$(CURRENT_SOURCE_DIR)/f16.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear -1:1 -r urand -0x1p-8:0x1p-8 -r linear 1:15 \
    -r linear -15:-1 -r linear -14:16 -r linear 15:16 \
    -r linear 16:32 -r linear -25:-14 -r linear -50:-25 \
    -r urand : \
    -o $@

$(CURRENT_SOURCE_DIR)/bf16.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear -1:1 -r urand -0x1p-8:0x1p-8 -r linear 1:32 \
    -r linear -32:-1 -r linear -126:128 -r linear 127:128 \
    -r linear 128:256 -r linear -134:-126 -r linear -268:-134 \
    -r urand : \
    -o $@
