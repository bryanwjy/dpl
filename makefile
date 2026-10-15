# Copyright 2025-2026 Bryan Wong

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))
ROOT_DIR := $(MKFILE_DIR)
OUTPUT_DIR := $(ROOT_DIR)/build
TOOLS_DIR := $(ROOT_DIR)/tools
INCLUDE_DIR := $(ROOT_DIR)/include
CLEAN_MANIFEST := $(OUTPUT_DIR)/clean_manifest.txt

CXX := clang++-21
SCAN_DEPS := clang-scan-deps-21

# Hard-code for now
CXX_FAMILY := clang

ifeq ($(filter -std=%,$(CXXFLAGS)),)
CXXFLAGS += -std=c++23
endif

# When on clang >= 22; until someone fixes immintrin.h and issues with implicit function template instantiations
# add -Wno-TU-local-entity-exposure -Wno-reference-tu-local-entity-in-other-tu to reduce noise
CXXFLAGS += -march=native -fvisibility=hidden -fvisibility-inlines-hidden
CPPFLAGS += -I$(INCLUDE_DIR)

define replace_if_different
set -e; \
mkdir -p '$(@D)'; \
tmpfile=$$(mktemp); \
$(1) > "$$tmpfile"; \
if ! cmp -s "$$tmpfile" "$@"; then \
    mv "$$tmpfile" "$@"; \
else \
    rm "$$tmpfile"; \
fi
endef

needs_build = $(if $(_NEEDS_BUILD_PROBE),,$(shell $(MAKE) -q --no-print-directory _NEEDS_BUILD_PROBE=1 $1 >/dev/null 2>&1 || echo 1))
rwildcard = $(foreach d,$(wildcard $(1:=/*)),$(if $(wildcard $d/.),$(call rwildcard,$d,$2),$(filter $(subst *,%,$2),$d)))
rdirs    = $1 $(foreach d,$(wildcard $(1:=/*/.)),$(call rdirs,$(patsubst %/.,%,$d)))

empty :=
space := $(empty) $(empty)
# newline requiures two empty lines so that dereferencing it results in a single newline
define newline


endef

NO_INCLUDE_GOALS := clean
# Non-empty unless every goal on the command line is a clean-type goal
NEED_INCLUDES := $(if $(MAKECMDGOALS),$(filter-out $(NO_INCLUDE_GOALS),$(MAKECMDGOALS)),default)
safe_include = $(if $(NEED_INCLUDES),$(eval -include $1))

.PHONY: everything all clean FORCE
.SECONDEXPANSION:

ALL_TARGETS :=
ALL_SOURCES :=
PREPROCESSOR_RESPONSES =
COMPILE_RESPONSES = module
LINK_RESPONSES = link

all: $(CLEAN_MANIFEST) $(OUTPUT_DIR)/compile_commands.json everything

include $(ROOT_DIR)/test/dpl/test.mk
include $(ROOT_DIR)/test/mpfr/mpfr.mk
include $(ROOT_DIR)/test/support/support.mk
include $(ROOT_DIR)/modules/dpl.mk

$(OUTPUT_DIR)/%/: 
	@mkdir -p $@

$(OUTPUT_DIR)/: 
	@mkdir -p $@

$(CLEAN_MANIFEST): $(ALL_SOURCES) | $(OUTPUT_DIR)
	@$(file >$@,$(foreach f,$(sort $(ALL_TARGETS)),$(f)$(newline)))

$(OUTPUT_DIR)/compile_commands.json: $(filter %.jcmd, $(ALL_TARGETS))
	@jq -s '.' $^ > $@

everything: $(ALL_TARGETS)
	@

clean:
	@if [ -s $(CLEAN_MANIFEST) ]; then \
	    xargs rm -f < $(CLEAN_MANIFEST) && rm $(CLEAN_MANIFEST); \
	else \
		rm -f $(ALL_TARGETS); \
	fi
