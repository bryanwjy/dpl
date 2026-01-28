# Copyright 2025-2026 Bryan Wong

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))

ROOT_DIR := $(MKFILE_DIR)
INCLUDE_DIR := $(ROOT_DIR)/include
SRC_DIR := $(ROOT_DIR)/src
TOOLS_DIR := $(ROOT_DIR)/tools
MODULES_DIR := $(ROOT_DIR)/modules
OUTPUT_DIR := $(ROOT_DIR)/build

CXX := clang++-21

ifneq ($(filter -std=%,$(CXXFLAGS)),)
# already has -std= → do nothing
else
CXXFLAGS += -std=c++23
endif

CPPFLAGS += -I$(INCLUDE_DIR)

MODULE_SRCS := $(shell find $(MODULES_DIR) -name '*.cppm')
TU_SRCS := $(shell find $(SRC_DIR) -name '*.cpp')

JDEP_TARGETS := $(addsuffix .jdep, $(MODULE_SRCS:$(MODULES_DIR)/%=%))
ARG_TARGETS := $(addsuffix .args, $(MODULE_SRCS:$(MODULES_DIR)/%=%))
BMI_TARGETS := $(addsuffix .pcm, $(MODULE_SRCS:$(MODULES_DIR)/%=%))
BMID_TARGETS := $(addsuffix .d, $(MODULE_SRCS:$(MODULES_DIR)/%=%))
BMIPP_TARGETS := $(addsuffix .i, $(MODULE_SRCS:$(MODULES_DIR)/%=%))
BMIJCMD_TARGETS := $(addsuffix .jcmd, $(MODULE_SRCS:$(MODULES_DIR)/%=%))

OBJ_TARGETS := $(addsuffix .o, $(TU_SRCS:$(SRC_DIR)/%=%))
OBJD_TARGETS := $(addsuffix .d, $(TU_SRCS:$(SRC_DIR)/%=%))
OBJPP_TARGETS := $(addsuffix .i, $(TU_SRCS:$(SRC_DIR)/%=%))
OBJJCMD_TARGETS := $(addsuffix .jcmd, $(TU_SRCS:$(SRC_DIR)/%=%))

JCMD_TARGETS := $(BMIJCMD_TARGETS) $(OBJJCMD_TARGETS)

ALL_TARGETS := $(JDEP_TARGETS) $(ARG_TARGETS) $(BMI_TARGETS) $(BMID_TARGETS) \
$(BMIPP_TARGETS) $(OBJ_TARGETS) $(OBJD_TARGETS) $(OBJPP_TARGETS) $(JCMD_TARGETS)

BMI_DIR := $(OUTPUT_DIR)/modules
OBJ_DIR := $(OUTPUT_DIR)/obj


.PHONY: compile.command link.command jmap jgraph FORCE print all compile_commands $(ALL_TARGETS)

.SECONDARY:  %/ $(OUTPUT_DIR)/%.d $(OUTPUT_DIR)/%.i $(OUTPUT_DIR)/%.command \
$(OUTPUT_DIR)/%.stamp  $(OUTPUT_DIR)/%.jcmd $(BMI_DIR)/%.jdep $(BMI_DIR)/%.args \
$(BMI_DIR)/%jmap $(OUTPUT_DIR)/compile_commands.json

all: $(BMI_TARGETS) $(OUTPUT_DIR)/compile_commands.json
	@

$(OUTPUT_DIR)/env.stamp: FORCE
	@mkdir -p '$(@D)'
	@tmp=$@.tmp; \
	printf "%s\n" \
	  "CXX=$(CXX)" \
	  "CPPFLAGS=$(CPPFLAGS)" \
	  "CXXFLAGS=$(CXXFLAGS)" \
	  "LDFLAGS=$(LDFLAGS)" \
	  "LDLIBS=$(LDLIBS)" \
	| sha256sum > $$tmp && \
	cmp -s $$tmp $@ || mv $$tmp $@

$(OUTPUT_DIR)/compile.command: makefile $(OUTPUT_DIR)/env.stamp
	@echo "$(CXX) $(CPPFLAGS) $(CXXFLAGS)" > $@

$(OUTPUT_DIR)/link.command: makefile $(OUTPUT_DIR)/env.stamp
	@echo "$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)" > $@

compile.command: $(OUTPUT_DIR)/compile.command
	@

link.command: $(OUTPUT_DIR)/link.command
	@

jmap: $(BMI_DIR)/dpl.jmap
	@

jgraph: $(BMI_DIR)/dpl.jgraph
	@

compile_commands: $(OUTPUT_DIR)/compile_commands.json
	@

$(BMI_TARGETS):%.cppm.pcm: $(BMI_DIR)/%.cppm.pcm
	@

$(ARG_TARGETS):%.cppm.args: $(BMI_DIR)/%.cppm.args
	@

$(JDEP_TARGETS):%.cppm.jdep: $(BMI_DIR)/%.cppm.jdep
	@

$(BMI_DIR)/%.cppm.pcm: $(MODULES_DIR)/%.cppm $(BMI_DIR)/%.cppm.args
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.pcm=.d)' -MT '$@' --precompile $< -o '$@' @$(BMI_DIR)/$*.cppm.args

-include $(addprefix $(BMI_DIR)/,$(BMID_TARGETS))

$(BMI_DIR)/%.cppm.jcmd: $(OUTPUT_DIR)/compile.command $(BMI_DIR)/%.cppm.args
	@echo "{ \"directory\": \"$(BMI_DIR)\", \"command\": \"$(CXX) $(CPPFLAGS) $(CXXFLAGS) --precompile $(MODULES_DIR)/$*.cppm -o $*.cppm.pcm @$*.cppm.args\", \"file\": \"$(MODULES_DIR)/$*.cppm\" }" > $@

$(BMI_DIR)/%.cppm.jdep: $(MODULES_DIR)/%.cppm
	@mkdir -p '$(@D)'
	@clang-scan-deps-21 -format=p1689 -o $@ -- $(CXX) $(CPPFLAGS) $(CXXFLAGS) --precompile $< -o $(@:.jdep=.pcm)

$(BMI_DIR)/dpl.jmap: $(TOOLS_DIR)/jdeps-to-jmap.jq $(addprefix $(BMI_DIR)/,$(JDEP_TARGETS)) 
	@jq -s -f $^ > $@

$(BMI_DIR)/dpl.jgraph: $(TOOLS_DIR)/jdeps-to-jgraph.jq $(addprefix $(BMI_DIR)/,$(JDEP_TARGETS)) 
	@jq -s -f $^ > $@

$(OUTPUT_DIR)/compile_commands.json: $(addprefix $(BMI_DIR)/,$(BMIJCMD_TARGETS)) $(addprefix $(OBJ_DIR)/,$(OBJJCMD_TARGETS)) 
	@jq -s '.' $^ > $@

$(BMI_DIR)/%.cppm.d: $(BMI_DIR)/%.cppm.jdep $(BMI_DIR)/dpl.jmap $(TOOLS_DIR)/jdep-to-d.jq
	@jq -r --slurpfile jmap $(BMI_DIR)/dpl.jmap -f $(TOOLS_DIR)/jdep-to-d.jq $< > $@

$(BMI_DIR)/%.cppm.args: $(BMI_DIR)/dpl.jmap $(BMI_DIR)/dpl.jgraph $(BMI_DIR)/%.cppm.jdep
	@jq -r --slurpfile jmap $(BMI_DIR)/dpl.jmap \
     --slurpfile jgraph $(BMI_DIR)/dpl.jgraph \
     --arg target "$(shell jq -r '.rules[0].provides[0]["logical-name"]' $(BMI_DIR)/$*.cppm.jdep)" \
     -f $(TOOLS_DIR)/jgraph-to-args.jq \
     $(BMI_DIR)/$*.cppm.jdep > $@

