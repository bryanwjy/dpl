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

ifneq ($(filter -std=%,$(CXXFLAGS)),)
	# Extract the actual standard (e.g. -std=c++20 → c++20)
    CXXSTD := $(patsubst -std=%,%,$(filter -std=%,$(CXXFLAGS)))
    # Error if the standard is less than c++23
    ifneq ($(filter c++23 c++2b,$(CXXSTD)),)
        # good: >= c++23
    else
        $(error CXXFLAGS specifies $(CXXSTD), which is less than c++23)
    endif
else
CXXFLAGS += -std=c++23
endif

CXXFLAGS += -msse4.2 -mfma
CPPFLAGS += -I$(INCLUDE_DIR)

CPPM_SRCS := $(shell find $(MODULES_DIR) -name '*.cppm') $(shell find $(SRC_DIR) -name '*.cppm') \
$(shell find $(TEST_DIR) -name '*.cppm')
CPP_SRCS := $(shell find $(SRC_DIR) -name '*.cpp') $(shell find $(TEST_DIR) -name '*.cpp')
SRCS := $(CPPM_SRCS) $(CPP_SRCS)

BMI_TARGETS := $(addsuffix .pcm, $(CPPM_SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
BMID_TARGETS := $(addsuffix .d, $(CPPM_SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))

JDEP_TARGETS := $(addsuffix .jdep, $(SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
ARG_TARGETS := $(addsuffix .args, $(SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
OBJ_TARGETS := $(addsuffix .o, $(SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
OBJD_TARGETS := $(addsuffix .d, $(SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
JCMD_TARGETS := $(addsuffix .jcmd, $(SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
PRE_TARGETS := $(addsuffix .pre, $(SRCS:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))

ALL_TARGETS := $(BMI_TARGETS) $(BMID_TARGETS) \
$(JDEP_TARGETS) $(ARG_TARGETS) $(OBJ_TARGETS) \
$(OBJD_TARGETS) $(JCMD_TARGETS)

TEST_OBJ_TARGETS := $(filter %.pass.cpp.o,$(OBJ_TARGETS))
PASS_OBJ_TARGETS := $(filter-out %.compile.pass.cpp.o,$(TEST_OBJ_TARGETS))
PASS_EXES := $(patsubst %.cpp.o,%,$(PASS_OBJ_TARGETS))
TEST_CRC := $(addsuffix .crc, $(PASS_EXES))

.PHONY: all clean jmap jgraph compile_commands FORCE

.PRECIOUS: $(OUTPUT_DIR)/%.d $(OUTPUT_DIR)/%.command \
$(OUTPUT_DIR)/%.stamp $(OUTPUT_DIR)/%.jdep $(OUTPUT_DIR)/%.args \
$(OUTPUT_DIR)/jmap.json $(OUTPUT_DIR)/jgraph.json $(OUTPUT_DIR)/compile_commands.json

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

all : $(TEST_CRC) $(PASS_EXES) $(OUTPUT_DIR)/compile_commands.json
	@

$(OUTPUT_DIR)/%.pass.crc: $(OUTPUT_DIR)/%.pass
	@$< && echo "\033[0;34mTEST\033[0m $*: \033[0;32mSUCCESS\033[0m" && \
	cksum $< > $@ || { echo "TEST $(OUTPUT_DIR)/%,%,$<): \033[0;31mFAILED\033[0m" && rm -f $@; exit 1; }

$(OUTPUT_DIR)/%.pass: $(OUTPUT_DIR)/%.pass.cpp.o $(OUTPUT_DIR)/link.command
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $< $(LDLIBS) -o '$@'

clean:
	@rm -f $(OUTPUT_DIR)/**/*.o $(OUTPUT_DIR)/**/*.pcm $(OUTPUT_DIR)/**/*.jdep \
	$(OUTPUT_DIR)/**/*.d  $(OUTPUT_DIR)/**/*.args $(OUTPUT_DIR)/jgraph.json $(OUTPUT_DIR)/jmap.json \
	$(OUTPUT_DIR)/compile.command $(OUTPUT_DIR)/link.command $(OUTPUT_DIR)/env.stamp

$(OUTPUT_DIR)/env.stamp: FORCE
	$(call replace_if_different, printf "%s\n" \
	  "CXX=$(CXX)" \
	  "CPPFLAGS=$(CPPFLAGS)" \
	  "CXXFLAGS=$(CXXFLAGS)" \
	  "LDFLAGS=$(LDFLAGS)" \
	  "LDLIBS=$(LDLIBS)" \
	| sha256sum)

$(OUTPUT_DIR)/compile.command: makefile $(OUTPUT_DIR)/env.stamp
	@echo "$(CXX) $(CPPFLAGS) $(CXXFLAGS)" > $@

$(OUTPUT_DIR)/link.command: makefile $(OUTPUT_DIR)/env.stamp
	@echo "$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)" > $@

jmap: $(OUTPUT_DIR)/jmap.json
	@

jgraph: $(OUTPUT_DIR)/jgraph.json
	@

compile_commands: $(OUTPUT_DIR)/compile_commands.json
	@

$(OUTPUT_DIR)/jmap.json: $(TOOLS_DIR)/jdeps-to-jmap.jq $(JDEP_TARGETS)
	@jq -s -f $^ > $@

$(OUTPUT_DIR)/jgraph.json: $(TOOLS_DIR)/jdeps-to-jgraph.jq $(JDEP_TARGETS)
	@jq -s -f $^ > $@

$(OUTPUT_DIR)/compile_commands.json: $(JCMD_TARGETS)
	@jq -s '.' $^ > $@

$(OUTPUT_DIR)/%.args: $(OUTPUT_DIR)/%.jdep $(OUTPUT_DIR)/jmap.json $(OUTPUT_DIR)/jgraph.json
	$(call replace_if_different, \
	jq -r --slurpfile jmap $(OUTPUT_DIR)/jmap.json \
    --slurpfile jgraph $(OUTPUT_DIR)/jgraph.json \
    -f $(TOOLS_DIR)/jgraph-to-args.jq \
    $<)

$(PRE_TARGETS):%.pre: %.jdep $(OUTPUT_DIR)/jmap.json $(TOOLS_DIR)/jdep-to-d.jq
	$(call replace_if_different, jq -r --slurpfile jmap $(OUTPUT_DIR)/jmap.json -f $(TOOLS_DIR)/jdep-to-d.jq $<)

-include $(PRE_TARGETS)

$(OUTPUT_DIR)/%.cppm.pcm: $(ROOT_DIR)/%.cppm $(OUTPUT_DIR)/%.cppm.args | $(OUTPUT_DIR)/%.cppm.pre
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.pcm=.d)' -MT '$@' --precompile $< -o '$@' @$(OUTPUT_DIR)/$*.cppm.args

-include $(BMID_TARGETS)

$(OUTPUT_DIR)/%.cppm.jcmd: $(OUTPUT_DIR)/%.cppm.args $(OUTPUT_DIR)/compile.command $(TOOLS_DIR)/generate-jcmd.jq
	@jq -n \
	--arg directory '$(OUTPUT_DIR)' \
	--rawfile args '$(OUTPUT_DIR)/$*.cppm.args' \
	--arg command '$(CXX) $(CPPFLAGS) $(CXXFLAGS) -x c++ -c $*.cppm.pcm -o $*.cppm.o' \
	--arg file '$(ROOT_DIR)/$*.cppm' \
	-f $(TOOLS_DIR)/generate-jcmd.jq > $@

$(OUTPUT_DIR)/%.cpp.jcmd: $(OUTPUT_DIR)/%.cpp.args $(OUTPUT_DIR)/compile.command $(TOOLS_DIR)/generate-jcmd.jq
	@jq -n \
	--arg directory '$(OUTPUT_DIR)' \
	--rawfile args '$(OUTPUT_DIR)/$*.cpp.args' \
	--arg command '$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(ROOT_DIR)/$*.cpp -o $*.cpp.o' \
	--arg file '$(ROOT_DIR)/$*.cpp' \
	-f $(TOOLS_DIR)/generate-jcmd.jq > $@

$(OUTPUT_DIR)/%.cppm.jdep: $(ROOT_DIR)/%.cppm
	$(call replace_if_different, clang-scan-deps-21 -format=p1689 -- $(CXX) $(CPPFLAGS) $(CXXFLAGS) --precompile $< -o $(@:.jdep=.pcm))

$(OUTPUT_DIR)/%.cpp.jdep: $(ROOT_DIR)/%.cpp
	$(call replace_if_different, clang-scan-deps-21 -format=p1689 -- $(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $(@:.jdep=.o))

$(OUTPUT_DIR)/%.cppm.o: $(OUTPUT_DIR)/%.cppm.pcm $(OUTPUT_DIR)/%.cppm.args
	@mkdir -p '$(@D)'
	@$(CXX) $(CXXFLAGS) -c $< -o '$@' @$(OUTPUT_DIR)/$*.cppm.args

$(OUTPUT_DIR)/%.cpp.o: $(ROOT_DIR)/%.cpp $(OUTPUT_DIR)/%.cpp.args | $(OUTPUT_DIR)/%.cpp.pre
	@mkdir -p '$(@D)'
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.o=.d)' -MT '$@' -c $< -o '$@' @$(OUTPUT_DIR)/$*.cpp.args

-include $(OBJD_TARGETS)
