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
    -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 -r linear 1:32  \
    -r linear -32:-1 -r linear -87.3:88.72 -r linear 87.7:88.72 \
    -r linear 88.73:178 -r linear -103.97:-87.34 -r linear -208:-103.98 \
    -r urand : \
    -o $@

$(CURRENT_SOURCE_DIR)/f64.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 -r linear 1:32 \
    -r linear -32:-1 -r linear -708.3:709.78 -r linear 708.7:709.78 \
    -r linear 709.79:1420 -r linear -745.13:-708.4 -r linear -1490:-745.14 \
    -r urand : \
    -o $@

$(CURRENT_SOURCE_DIR)/f16.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 -r linear 1:8 \
    -r linear -8:-1 -r linear -9.7:11.08 -r linear 10:11.08 \
    -r linear 11.1:22 -r linear -17.32:-9.71 -r linear -35:-17.33 \
    -r urand : \
    -o $@

$(CURRENT_SOURCE_DIR)/bf16.h:$(CURRENT_SOURCE_DIR)/%.h:
	@$(MPFR_BIN) $(CURRENT_FUNCTION) $* -n $(COUNT) --seed $(SEED) \
    -r linear -0.5:0.5 -r urand -0x1p-8:0x1p-8 -r linear 1:32 \
    -r linear -32:-1 -r linear -87.3:88.5 -r linear 87.5:88.5 \
    -r linear 89:178 -r linear -92.88:-87.34 -r linear -186:-92.89 \
    -r urand : \
    -o $@
