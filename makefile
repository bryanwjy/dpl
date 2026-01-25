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

OBJ_TARGETS := $(addsuffix .o, $(TU_SRCS:$(SRC_DIR)/%=%))
OBJD_TARGETS := $(addsuffix .d, $(TU_SRCS:$(SRC_DIR)/%=%))
OBJPP_TARGETS := $(addsuffix .i, $(TU_SRCS:$(SRC_DIR)/%=%))

ALL_TARGETS := $(BMI_TARGETS) $(ARG_TARGETS) $(JDEP_TARGETS) $(BMID_TARGETS) $(BMIPP_TARGETS) $(OBJ_TARGETS) $(OBJD_TARGETS) $(OBJPP_TARGETS)

BMI_DIR := $(OUTPUT_DIR)/modules
OBJ_DIR := $(OUTPUT_DIR)/obj

.PHONY: compile.command link.command dpl.jmap dpl.graph FORCE print all $(ALL_TARGETS)

.SECONDARY:  %/ $(OUTPUT_DIR)/%.d $(OUTPUT_DIR)/%.i $(OUTPUT_DIR)/%.command $(OUTPUT_DIR)/%.stamp $(BMI_DIR)/%.jdep $(BMI_DIR)/%.args $(BMI_DIR)/%jmap

all: $(ARG_TARGETS)
	@

$(OUTPUT_DIR)/env.stamp: FORCE | $(OUTPUT_DIR)/
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

$(OUTPUT_DIR)/compile.command: makefile $(ENV_STAMP) | $(OUTPUT_DIR)/
	@echo "$(CXX) $(CPPFLAGS) $(CXXFLAGS)" > $@

$(OUTPUT_DIR)/link.command: makefile $(ENV_STAMP) | $(OUTPUT_DIR)/
	@echo "$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)" > $@

compile.command: $(OUTPUT_DIR)/compile.command
	@cat $<

link.command: $(OUTPUT_DIR)/link.command
	@cat $<

dpl.jmap: $(BMI_DIR)/dpl.jmap
	@

dpl.jgraph: $(BMI_DIR)/dpl.jgraph
	@

$(BMI_TARGETS):%.cppm.pcm: $(BMI_DIR)/%.cppm.pcm
	@

$(BMID_TARGETS):%.cppm.d: $(BMI_DIR)/%.cppm.d
	@

$(ARG_TARGETS):%.cppm.args: $(BMI_DIR)/%.cppm.args
	@

$(JDEP_TARGETS):%.cppm.jdep: $(BMI_DIR)/%.cppm.jdep
	@

$(BMI_DIR)/%.cppm.pcm: $(MODULES_DIR)/%.cppm $(BMI_DIR)/%.cppm.args
	@

$(BMI_DIR)/%.cppm.jdep: $(MODULES_DIR)/%.cppm
	@mkdir -p '$(@D)'
	@clang-scan-deps-21 -format=p1689 -o $@ -- $(CXX) $(CPPFLAGS) $(CXXFLAGS) --precompile $< -o $(@:.jdep=.pcm)

$(BMI_DIR)/dpl.jmap: $(TOOLS_DIR)/jdeps-to-jmap.jq $(addprefix $(BMI_DIR)/,$(JDEP_TARGETS)) 
	@jq -s -f $^ > $@

$(BMI_DIR)/dpl.jgraph: $(TOOLS_DIR)/jdeps-to-jgraph.jq $(addprefix $(BMI_DIR)/,$(JDEP_TARGETS)) 
	@jq -s -f $^ > $@

$(BMI_DIR)/%.cppm.d: $(BMI_DIR)/%.cppm.jdep $(BMI_DIR)/dpl.jmap $(TOOLS_DIR)/jdep-to-d.jq
	@jq -r --slurpfile jmap $(BMI_DIR)/dpl.jmap -f $(TOOLS_DIR)/jdep-to-d.jq $< > $@

-include $(addprefix $(BMI_DIR)/,$(BMID_TARGETS))

$(BMI_DIR)/%.cppm.args: $(BMI_DIR)/dpl.jmap $(BMI_DIR)/dpl.jgraph $(BMI_DIR)/%.cppm.jdep
	@jq -r --slurpfile jmap $(BMI_DIR)/dpl.jmap \
     --slurpfile jgraph $(BMI_DIR)/dpl.jgraph \
     --arg target "$(shell jq -r '.rules[0].provides[0]["logical-name"]' $(BMI_DIR)/$*.cppm.jdep)" \
     -f $(TOOLS_DIR)/jgraph-to-args.jq \
     $(BMI_DIR)/$*.cppm.jdep > $@

