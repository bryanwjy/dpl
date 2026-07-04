# Copyright 2025-2026 Bryan Wong

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))

ROOT_DIR := $(MKFILE_DIR)
INCLUDE_DIR := $(ROOT_DIR)/include
SRC_DIR := $(ROOT_DIR)/src
TEST_DIR := $(ROOT_DIR)/test
TOOLS_DIR := $(ROOT_DIR)/tools
MODULES_DIR := $(ROOT_DIR)/modules
OUTPUT_DIR := $(ROOT_DIR)/build

CXX := clang++-21
# Hard-code for now
CXX_FAMILY := clang

ifeq ($(filter -std=%,$(CXXFLAGS)),)
CXXFLAGS += -std=c++23
endif

CXXFLAGS += -mfma -mavx2 -fvisibility=hidden -fvisibility-inlines-hidden
CPPFLAGS += -I$(INCLUDE_DIR)

ALL_SOURCES := $(shell find $(MODULES_DIR) $(SRC_DIR) $(TEST_DIR) -type f \( -name '*.cpp' -o -name '*.cppm' \) 2>/dev/null)
CPP_SOURCES := $(filter %.cpp,$(ALL_SOURCES))
MODULE_SOURCES := $(filter %.cppm,$(ALL_SOURCES))
COMPILE_SOURCES := $(CPP_SOURCES) $(filter-out $(MODULES_DIR)/%,$(MODULE_SOURCES))
PASS_SOURCES := $(filter %.pass.cpp,$(CPP_SOURCES))
# TODO Death tests
# VERIFY_SOURCES := $(filter %.verify.cpp,$(CPP_SOURCES))

# Include subdirs

# Precompiled modules
PCM_TARGETS := $(addsuffix .pcm, $(MODULE_SOURCES:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
# Object files
OBJ_TARGETS := $(addsuffix .o, $(COMPILE_SOURCES:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
MOBJ_TARGETS := $(filter %.cppm.o,$(OBJ_TARGETS))
OBJ_TARGETS := $(filter-out %.cppm.o,$(OBJ_TARGETS))

OUTPUT_STEMS := $(ALL_SOURCES:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%)
# Module response files
MRSP_TARGETS := $(addsuffix .mrsp,$(OUTPUT_STEMS))
# Compile command files: For static analyses
JCMD_TARGETS := $(addsuffix .jcmd,$(OUTPUT_STEMS))
# Scan command files: For dependency scanning
JSCAN_TARGETS := $(addsuffix .jscan,$(OUTPUT_STEMS))

OUTPUT_STEMS := $(PASS_SOURCES:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%)
# Test response files
TRSP_TARGETS := $(addsuffix .trsp, $(OUTPUT_STEMS))
# Directive files (Extra compile flags defined in the test file)
JDIR_TARGETS := $(addsuffix .jdir, $(OUTPUT_STEMS))
# Link response files
LRSP_TARGETS := $(addsuffix .lrsp, $(OUTPUT_STEMS))
OUTPUT_STEMS :=

COMPILE_PASS_TARGETS := $(filter %.compile.pass.cpp.o,$(OBJ_TARGETS))
PASS_TARGETS := $(filter %.pass.cpp.o,$(OBJ_TARGETS))
PASS_TARGETS := $(filter-out %.compile.pass.cpp.o,$(PASS_TARGETS))
PASS_TARGETS := $(patsubst %.cpp.o,%,$(PASS_TARGETS))

# Dependency makefiles
DEP_FILES := $(addsuffix .d, $(ALL_SOURCES:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))

ALL_TARGETS := $(PCM_TARGETS) $(MRSP_TARGETS) $(JCMD_TARGETS) $(JSCAN_TARGETS) \
$(OBJ_TARGETS) $(LRSP_TARGETS) $(MOBJ_TARGETS) $(TRSP_TARGETS) $(JDIR_TARGETS) $(PASS_TARGETS)

COMPILE_PASS_CRC := $(patsubst %.cpp.o,%.crc,$(COMPILE_PASS_TARGETS))
PASS_CRC := $(addsuffix .crc,$(PASS_TARGETS))
TEST_CRC := $(PASS_CRC) $(COMPILE_PASS_CRC)
TEST_SUBDIRS := $(shell find $(TEST_DIR) -type d)
TEST_SUBDIRS := $(TEST_SUBDIRS:$(ROOT_DIR)/%=%)
TEST_ALIAS := $(TEST_CRC:$(OUTPUT_DIR)/%.crc=%)

BUILD_JSON := $(OUTPUT_DIR)/jmap.json $(OUTPUT_DIR)/jgraph.json $(OUTPUT_DIR)/compile_commands.json \
$(OUTPUT_DIR)/module_dependencies.json $(OUTPUT_DIR)/scan_commands.json
BUILD_TXT := $(OUTPUT_DIR)/candidate_flags.txt $(OUTPUT_DIR)/module_implementations.txt
BUILD_FILES := $(BUILD_JSON) $(BUILD_TXT)
BUILD_JSON_ALIAS := $(basename $(notdir $(BUILD_JSON)))
BUILD_TXT_ALIAS := $(basename $(notdir $(BUILD_TXT)))
MODULE_ALIAS := $(basename $(MODULE_SOURCES:$(ROOT_DIR)/%=%))

.PHONY: all clean parallel_probes FORCE $(BUILD_JSON_ALIAS) $(BUILD_TXT_ALIAS) $(TEST_ALIAS) $(TEST_SUBDIRS) $(MODULE_ALIAS)

define replace_if_different
@mkdir -p '$(@D)'
@tmpfile=$$(mktemp); \
$(1) > "$$tmpfile"; \
if ! cmp -s "$$tmpfile" "$@"; then \
    mv "$$tmpfile" "$@"; \
else \
    rm "$$tmpfile"; \
fi
endef

define subdir_to_crc
$(patsubst $(ROOT_DIR)/%.cpp,$(OUTPUT_DIR)/%.crc,$(wildcard $(ROOT_DIR)/$(1)/*.pass.cpp)) \
$(patsubst $(ROOT_DIR)/%.cpp,$(OUTPUT_DIR)/%.crc,$(wildcard $(ROOT_DIR)/$(1)/**/*.pass.cpp))
endef

all: $(TEST_CRC) $(ALL_TARGETS) $(BUILD_FILES)
	@

clean:
	@rm -f $(TEST_CRC) $(ALL_TARGETS) $(BUILD_FILES)

$(OUTPUT_DIR)/env.stamp: FORCE
	$(call replace_if_different, printf "%s\n" \
	  "CXX=$(CXX)" \
	  "CPPFLAGS=$(CPPFLAGS)" \
	  "CXXFLAGS=$(CXXFLAGS)" \
	  "LDFLAGS=$(LDFLAGS)" \
	  "LDLIBS=$(LDLIBS)")

$(OUTPUT_DIR)/compile.command: makefile $(OUTPUT_DIR)/env.stamp
	@$(call replace_if_different, echo "$(CXX) $(CPPFLAGS) $(CXXFLAGS)")

$(OUTPUT_DIR)/link.command: makefile $(OUTPUT_DIR)/env.stamp
	@$(call replace_if_different, echo "$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)")

$(PASS_CRC):$(OUTPUT_DIR)/%.crc: $(OUTPUT_DIR)/%
	@$< && echo "\033[0;34mTEST\033[0m $*: \033[0;32mSUCCESS\033[0m" && \
	cksum $< > $@ || { echo "TEST $*): \033[0;31mFAILED\033[0m" && rm -f $@; exit 1; }

$(COMPILE_PASS_CRC):$(OUTPUT_DIR)/%.crc: $(OUTPUT_DIR)/%.cpp.o
	@echo "\033[0;34mCOMPILE\033[0m $*: \033[0;32mSUCCESS\033[0m" && \
	cksum $< > $@ || { echo "\033[0;34mCOMPILE\033[0m $*: \033[0;31mFAILED\033[0m"; exit 1; }

$(PASS_TARGETS):%.pass: %.pass.cpp.o %.pass.cpp.lrsp $(OUTPUT_DIR)/link.command
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o '$@' $< $(LDLIBS) @$*.pass.cpp.lrsp

$(OUTPUT_DIR)/jmap.json: $(TOOLS_DIR)/jdeps-to-jmap.jq $(OUTPUT_DIR)/module_dependencies.json
	@jq -f $(TOOLS_DIR)/jdeps-to-jmap.jq -s $(OUTPUT_DIR)/module_dependencies.json > $@

$(OUTPUT_DIR)/jgraph.json: $(TOOLS_DIR)/graph.jq $(OUTPUT_DIR)/module_dependencies.json
	@jq -f $^ > $@

$(OUTPUT_DIR)/compile_commands.json: $(JCMD_TARGETS)
	@jq -s '.' $^ > $@

$(OUTPUT_DIR)/scan_commands.json: $(JSCAN_TARGETS)
	@jq -s '.' $^ > $@

$(OUTPUT_DIR)/module_dependencies.json: $(OUTPUT_DIR)/scan_commands.json $(ALL_SOURCES)
	@clang-scan-deps-21 -format=p1689 -compilation-database=$< -o $@

$(OUTPUT_DIR)/module_implementations.txt: FORCE
	@mkdir -p '$(@D)'
	@$(file >$@.tmp,$(subst $(space),$(newline),$(strip $(MOBJ_TARGETS))))
	@cmp -s $@.tmp $@ 2>/dev/null && rm $@.tmp || mv $@.tmp $@

$(OUTPUT_DIR)/candidate_flags.txt: $(JDIR_TARGETS)
	@jq -r '.[] | .["compile-flags"] // [] | .[]' $^ | sort -u > $@

$(OUTPUT_DIR)/scan_barrier.mk: $(OUTPUT_DIR)/jmap.json $(OUTPUT_DIR)/module_dependencies.json $(TOOLS_DIR)/jdep-to-d.jq
	@jq -f $(TOOLS_DIR)/jdep-to-d.jq --slurpfile jmap $(OUTPUT_DIR)/jmap.json -r $(OUTPUT_DIR)/module_dependencies.json > $(OUTPUT_DIR)/scan_barrier.mk

-include $(OUTPUT_DIR)/scan_barrier.mk

$(MODULE_ALIAS):%: $(OUTPUT_DIR)/%.cppm.pcm
$(TEST_ALIAS):%.pass: $(OUTPUT_DIR)/%.pass
$(BUILD_JSON_ALIAS):%: $(OUTPUT_DIR)/%.json
$(BUILD_TXT_ALIAS):%: $(OUTPUT_DIR)/%.txt
.SECONDEXPANSION:
$(TEST_SUBDIRS):%: $$(call subdir_to_crc,%)
$(MOBJ_TARGETS):%.o: %.pcm
$(filter %.pass.cpp.o,$(OBJ_TARGETS)):%.o: %.trsp
$(filter %.pass.cpp.o,$(JCMD_TARGETS)):%.jcmd: %.trsp

$(PCM_TARGETS):$(OUTPUT_DIR)/%.cppm.pcm: $(ROOT_DIR)/%.cppm $(OUTPUT_DIR)/%.cppm.mrsp $(OUTPUT_DIR)/compile.command | $(OUTPUT_DIR)/scan_barrier.mk
	@tmpfile=$$(mktemp); \
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.pcm=.d)' -MT '$@' \
	  -fmodule-output="$$tmpfile" -fmodules-reduced-bmi -c $< -o '$(@:.pcm=.o)' @$(@:.pcm=.mrsp); \
	if cmp -s "$$tmpfile" '$@' 2>/dev/null; then \
	    rm "$$tmpfile"; \
	else \
	    mv "$$tmpfile" '$@'; \
	fi

$(OBJ_TARGETS):$(OUTPUT_DIR)/%.cpp.o: $(ROOT_DIR)/%.cpp $(OUTPUT_DIR)/%.cpp.mrsp $(OUTPUT_DIR)/compile.command | $(OUTPUT_DIR)/scan_barrier.mk
	@mkdir -p '$(@D)'
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.o=.d)' -MT '$@' -c $< -o '$@' @$(@:.o=.mrsp) $(if $(filter %.pass,$*),@$(@:.o=.trsp))

-include $(DEP_FILES)

$(JCMD_TARGETS):$(OUTPUT_DIR)/%.jcmd: $(OUTPUT_DIR)/%.mrsp $(OUTPUT_DIR)/compile.command $(TOOLS_DIR)/generate-jcmd.jq
	$(call replace_if_different, jq -f $(TOOLS_DIR)/generate-jcmd.jq -n \
	--arg directory '$(OUTPUT_DIR)' \
	--rawfile args '$(OUTPUT_DIR)/$*.mrsp' \
	$(if $(filter %.pass.cpp,$*),--rawfile trsp '$(OUTPUT_DIR)/$*.trsp') \
	--arg command '$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(if $(filter %.cppm,$*),-x c++ -fmodules-reduced-bmi )-c $(ROOT_DIR)/$* -o $*.o' \
	--arg file '$(ROOT_DIR)/$*' \
	--arg output '$(if $(filter modules/%.cppm,$*),$*.pcm,$*.o)')

$(JSCAN_TARGETS):$(OUTPUT_DIR)/%.jscan: $(ROOT_DIR)/% $(OUTPUT_DIR)/compile.command $(TOOLS_DIR)/generate-jcmd.jq
	$(call replace_if_different, jq -f $(TOOLS_DIR)/generate-jcmd.jq -n \
	--arg directory '$(OUTPUT_DIR)' \
	--arg command '$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(if $(filter %.cppm,$*),-x c++ -fmodules-reduced-bmi )-c $(ROOT_DIR)/$* -o $*.o' \
	--arg file '$(ROOT_DIR)/$*' \
	--arg output '$(if $(filter %.cppm,$*),$(@:.jscan=.pcm),$(@:.jscan=.o))')

$(JDIR_TARGETS):$(OUTPUT_DIR)/%.jdir: $(ROOT_DIR)/% $(TOOLS_DIR)/directives.awk
	$(call replace_if_different, $(TOOLS_DIR)/directives.awk -v family=$(CXX_FAMILY) $<)

$(MRSP_TARGETS):%.mrsp: $(OUTPUT_DIR)/jgraph.json $(OUTPUT_DIR)/jmap.json $(TOOLS_DIR)/module-response.jq
	$(call replace_if_different, $(TOOLS_DIR)/module-response.jq \
	--arg module $(if $(filter %.cppm,$*),$*.pcm,$*.o) \
	--slurpfile jmap $(OUTPUT_DIR)/jmap.json \
    -r $(OUTPUT_DIR)/jgraph.json)

$(LRSP_TARGETS):%.lrsp: $(OUTPUT_DIR)/jgraph.json $(OUTPUT_DIR)/jmap.json $(OUTPUT_DIR)/module_implementations.txt $(TOOLS_DIR)/link-response.jq
	$(call replace_if_different, jq -f $(TOOLS_DIR)/link-response.jq \
	--arg module '$*.o' \
	--slurpfile jmap $(OUTPUT_DIR)/jmap.json \
	--rawfile impl $(OUTPUT_DIR)/module_implementations.txt \
	-r $(OUTPUT_DIR)/jgraph.json)

$(TRSP_TARGETS):%.trsp: %.jdir $(OUTPUT_DIR)/supported_flags.txt
	@$(call replace_if_different, (jq -r '.[] | .["compile-flags"] // [] | .[]' $< | grep -Fxf $(OUTPUT_DIR)/supported_flags.txt - || true))

$(OUTPUT_DIR)/supported_flags.txt: $(OUTPUT_DIR)/candidate_flags.txt
	@mkdir -p $(OUTPUT_DIR)/probe
	@awk '{print $$0 > ("$(OUTPUT_DIR)/probe/" NR ".ptrn")}' $<
	@$(MAKE) --no-print-directory parallel_probes && cp $(OUTPUT_DIR)/probe/supported_flags.txt $(OUTPUT_DIR)/supported_flags.txt
	@echo "Supported DPL directive compile flags written to $(OUTPUT_DIR)/supported_flags.txt"

ifeq ($(MAKECMDGOALS),parallel_probes)
# Generate a list of 'valid' targets based on the 'ptrn' files we just created
PTRNS  := $(wildcard $(OUTPUT_DIR)/probe/*.ptrn)
VALIDS := $(patsubst %.ptrn,%.valid,$(PTRNS))

parallel_probes: $(OUTPUT_DIR)/probe/supported_flags.txt
	@echo "Probing DPL directive compile flags..."

$(OUTPUT_DIR)/probe/supported_flags.txt: $(VALIDS)
	@cat $(OUTPUT_DIR)/probe/*.valid > $@ 2>/dev/null || true

$(OUTPUT_DIR)/probe/%.valid: $(OUTPUT_DIR)/probe/%.ptrn
	@if echo "int main(){}" | $(CXX) -fsyntax-only @$< -x c++ - -o /dev/null >/dev/null 2>&1; then \
		cp $< $@; \
	else \
		touch $@; \
	fi
endif
