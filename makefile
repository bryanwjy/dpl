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

CXXFLAGS += -msse4.2 -mfma
CPPFLAGS += -I$(INCLUDE_DIR)

SRCS := $(shell find $(MODULES_DIR) $(SRC_DIR) $(TEST_DIR) -type f \( -name '*.cpp' -o -name '*.cppm' \) 2>/dev/null)
CPP_SRCS := $(filter %.cpp,$(SRCS))
CPPM_SRCS := $(filter %.cppm,$(SRCS))
COMPILE_SRCS := $(CPP_SRCS) $(filter $(SRC_DIR)/%.cppm,$(CPPM_SRCS))

BMI_TARGETS := $(addsuffix .pcm, $(CPPM_SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
MRSP_TARGETS += $(addsuffix .mrsp, $(SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
JCMD_TARGETS := $(addsuffix .jcmd, $(SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
JSCAN_TARGETS := $(addsuffix .jscan, $(SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
PRE_TARGETS := $(addsuffix .pre, $(SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
DEP_TARGETS := $(addsuffix .d, $(SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
OBJ_TARGETS := $(addsuffix .o, $(COMPILE_SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
TRSP_TARGETS := $(patsubst $(ROOT_DIR)/%,$(OUTPUT_DIR)/%.trsp,$(filter %.pass.cpp,$(CPP_SRCS)))

ALL_TARGETS := $(BMI_TARGETS) $(BMID_TARGETS) \
$(JDEP_TARGETS) $(MRSP_TARGETS) $(TRSP_TARGETS) $(OBJ_TARGETS) \
$(OBJD_TARGETS) $(JCMD_TARGETS)

TEST_OBJ_TARGETS := $(filter %.pass.cpp.o,$(OBJ_TARGETS))
TEST_OBJ_DIRECTIVES := $(patsubst %.cpp.o,%.jdir,$(TEST_OBJ_TARGETS))
PASS_OBJ_TARGETS := $(filter-out %.compile.pass.cpp.o,$(TEST_OBJ_TARGETS))
PASS_EXES := $(patsubst %.cpp.o,%,$(PASS_OBJ_TARGETS))
TEST_CRC := $(addsuffix .crc, $(PASS_EXES))

.PHONY: all clean jmap jgraph compile_commands module_dependencies parallel_probes FORCE

.PRECIOUS: $(DEP_TARGETS) $(MRSP_TARGETS) $(TRSP_TARGETS) $(OUTPUT_DIR)/jmap.json $(OUTPUT_DIR)/jgraph.json $(OUTPUT_DIR)/compile_commands.json

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

all: $(TEST_CRC) $(PASS_EXES) $(OUTPUT_DIR)/compile_commands.json $(OUTPUT_DIR)/candidate_flags.txt
	@

$(OUTPUT_DIR)/%.pass.crc: $(OUTPUT_DIR)/%.pass
	@$< && echo "\033[0;34mTEST\033[0m $*: \033[0;32mSUCCESS\033[0m" && \
	cksum $< > $@ || { echo "TEST $(OUTPUT_DIR)/%,%,$<): \033[0;31mFAILED\033[0m" && rm -f $@; exit 1; }

$(OUTPUT_DIR)/%.pass: $(OUTPUT_DIR)/%.pass.cpp.o $(OUTPUT_DIR)/link.command
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o '$@' $< $(LDLIBS)

clean:
	@rm -rf $(OUTPUT_DIR)

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

jmap: $(OUTPUT_DIR)/jmap.json
	@

jgraph: $(OUTPUT_DIR)/jgraph.json
	@

module_dependencies: $(OUTPUT_DIR)/module_dependencies.json
	@

compile_commands: $(OUTPUT_DIR)/compile_commands.json
	@

$(OUTPUT_DIR)/jmap.json: $(TOOLS_DIR)/jdeps-to-jmap.jq $(OUTPUT_DIR)/module_dependencies.json
	@jq -f $(TOOLS_DIR)/jdeps-to-jmap.jq -s $(OUTPUT_DIR)/module_dependencies.json > $@

$(OUTPUT_DIR)/jgraph.json: $(TOOLS_DIR)/graph.jq $(OUTPUT_DIR)/module_dependencies.json
	@jq -f $^ > $@

$(OUTPUT_DIR)/compile_commands.json: $(JCMD_TARGETS)
	@jq -s '.' $^ > $@

$(OUTPUT_DIR)/scan_commands.json: $(JSCAN_TARGETS)
	@jq -s '.' $^ > $@

$(OUTPUT_DIR)/module_dependencies.json: $(OUTPUT_DIR)/scan_commands.json
	@clang-scan-deps-21 -format=p1689 -compilation-database=$^ -o $@

$(OUTPUT_DIR)/%.cppm.mrsp: $(OUTPUT_DIR)/jgraph.json $(OUTPUT_DIR)/jmap.json $(TOOLS_DIR)/module-response.jq
	$(call replace_if_different, $(TOOLS_DIR)/module-response.jq \
	--arg module $(@:.mrsp=.pcm) \
	--slurpfile jmap $(OUTPUT_DIR)/jmap.json \
    -r $(OUTPUT_DIR)/jgraph.json)

$(OUTPUT_DIR)/%.cpp.mrsp: $(OUTPUT_DIR)/jgraph.json $(OUTPUT_DIR)/jmap.json $(TOOLS_DIR)/module-response.jq
	$(call replace_if_different, $(TOOLS_DIR)/module-response.jq \
	--arg module $(@:.mrsp=.o) \
	--slurpfile jmap $(OUTPUT_DIR)/jmap.json \
    -r $(OUTPUT_DIR)/jgraph.json)

$(OUTPUT_DIR)/scan_barrier.mk: $(OUTPUT_DIR)/jmap.json $(OUTPUT_DIR)/module_dependencies.json $(TOOLS_DIR)/jdep-to-d.jq
	@jq -f $(TOOLS_DIR)/jdep-to-d.jq --slurpfile jmap $(OUTPUT_DIR)/jmap.json -r $(OUTPUT_DIR)/module_dependencies.json > $(OUTPUT_DIR)/scan_barrier.mk

-include $(OUTPUT_DIR)/scan_barrier.mk

$(OUTPUT_DIR)/src/%.cppm.o: $(SRC_DIR)/%.cppm $(OUTPUT_DIR)/src/%.cppm.mrsp $(OUTPUT_DIR)/compile.command | $(OUTPUT_DIR)/scan_barrier.mk
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.o=.d)' -MT '$@' -fmodule-output=$(@:.o=.pcm) -fmodules-reduced-bmi -c $< -o '$@' @$(@:.o=.mrsp)

$(OUTPUT_DIR)/src/%.cppm.pcm: $(OUTPUT_DIR)/src/%.cppm.o
	@

$(OUTPUT_DIR)/modules/%.cppm.pcm: $(MODULES_DIR)/%.cppm $(OUTPUT_DIR)/modules/%.cppm.mrsp $(OUTPUT_DIR)/compile.command | $(OUTPUT_DIR)/scan_barrier.mk
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.pcm=.d)' -MT '$@' -fmodule-output=$@ -fmodules-reduced-bmi -c $< -o '$(@:.pcm=.o)' @$(@:.pcm=.mrsp)

$(OUTPUT_DIR)/%.cppm.jcmd: $(OUTPUT_DIR)/%.cppm.mrsp $(OUTPUT_DIR)/compile.command $(TOOLS_DIR)/generate-jcmd.jq
	@jq -f $(TOOLS_DIR)/generate-jcmd.jq -n \
	--arg directory '$(OUTPUT_DIR)' \
	--rawfile args '$(OUTPUT_DIR)/$*.cppm.mrsp' \
	--arg command '$(CXX) $(CPPFLAGS) $(CXXFLAGS) -x c++ -fmodules-reduced-bmi -c $(ROOT_DIR)/$*.cppm -o $*.cppm.o' \
	--arg file '$(ROOT_DIR)/$*.cppm' \
	--arg output '$(@:.jcmd=.pcm)' > $@

# Unused only defined for completeness
$(OUTPUT_DIR)/%.cpp.jcmd: $(OUTPUT_DIR)/%.cpp.mrsp $(OUTPUT_DIR)/compile.command $(TOOLS_DIR)/generate-jcmd.jq
	@jq -f $(TOOLS_DIR)/generate-jcmd.jq -n \
	--arg directory '$(OUTPUT_DIR)' \
	--rawfile args '$(OUTPUT_DIR)/$*.cpp.mrsp' \
	--arg command '$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(ROOT_DIR)/$*.cpp -o $*.cpp.o' \
	--arg file '$(ROOT_DIR)/$*.cpp' \
	--arg output '$(@:.jcmd=.o)' > $@

$(OUTPUT_DIR)/%.pass.cpp.jcmd: $(OUTPUT_DIR)/%.pass.cpp.mrsp $(OUTPUT_DIR)/%.pass.cpp.trsp $(OUTPUT_DIR)/compile.command $(TOOLS_DIR)/generate-jcmd.jq
	@jq -f $(TOOLS_DIR)/generate-jcmd.jq -n \
	--arg directory '$(OUTPUT_DIR)' \
	--rawfile mrsp '$(OUTPUT_DIR)/$*.pass.cpp.mrsp' \
	--rawfile trsp '$(OUTPUT_DIR)/$*.pass.cpp.trsp' \
	--arg command '$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(ROOT_DIR)/$*.cpp -o $*.cpp.o' \
	--arg file '$(ROOT_DIR)/$*.cpp' \
	--arg output '$(@:.jcmd=.o)' > $@

$(OUTPUT_DIR)/%.cppm.jscan: $(OUTPUT_DIR)/compile.command $(TOOLS_DIR)/generate-jcmd.jq $(ROOT_DIR)/%.cppm
	$(call replace_if_different, jq -f $(TOOLS_DIR)/generate-jcmd.jq -n \
		--arg directory '$(OUTPUT_DIR)' \
		--arg command '$(CXX) $(CPPFLAGS) $(CXXFLAGS) --precompile $(ROOT_DIR)/$*.cppm -o $*.cppm.pcm' \
		--arg file '$(ROOT_DIR)/$*.cppm' \
		--arg output '$(@:.jscan=.pcm)')

$(OUTPUT_DIR)/%.cpp.jscan: $(OUTPUT_DIR)/compile.command $(TOOLS_DIR)/generate-jcmd.jq | $(ROOT_DIR)/%.cpp
	$(call replace_if_different, jq -f $(TOOLS_DIR)/generate-jcmd.jq -n \
		--arg directory '$(OUTPUT_DIR)' \
		--arg command '$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(ROOT_DIR)/$*.cpp -o $*.cpp.o' \
		--arg file '$(ROOT_DIR)/$*.cppm' \
		--arg output '$(@:.jscan=.o)')

$(OUTPUT_DIR)/%.pass.jdir: $(ROOT_DIR)/%.pass.cpp $(TOOLS_DIR)/directives.awk
	$(call replace_if_different, $(TOOLS_DIR)/directives.awk -v family=$(CXX_FAMILY) $<)

$(OUTPUT_DIR)/candidate_flags.txt: $(TEST_OBJ_DIRECTIVES)
	@jq -r '.[] | .["compile-flags"] // [] | .[]' $^ | sort -u > $@

# Unused only defined for completeness
$(OUTPUT_DIR)/%.cpp.o: $(ROOT_DIR)/%.cpp $(OUTPUT_DIR)/%.cpp.mrsp $(OUTPUT_DIR)/compile.command | $(OUTPUT_DIR)/scan_barrier.mk
	@mkdir -p '$(@D)'
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.o=.d)' -MT '$@' -c $< -o '$@' @$(@:.o=.mrsp)

$(OUTPUT_DIR)/%.pass.cpp.o: $(ROOT_DIR)/%.pass.cpp $(OUTPUT_DIR)/%.pass.cpp.mrsp $(OUTPUT_DIR)/%.pass.cpp.trsp $(OUTPUT_DIR)/compile.command | $(OUTPUT_DIR)/scan_barrier.mk
	@mkdir -p '$(@D)'
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.o=.d)' -MT '$@' -c $< -o '$@' @$(@:.o=.mrsp) @$(@:.o=.trsp)

-include $(DEP_TARGETS)

$(OUTPUT_DIR)/%.pass.cpp.trsp: $(OUTPUT_DIR)/%.pass.jdir $(OUTPUT_DIR)/supported_flags.txt
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
