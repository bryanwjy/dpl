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
# already has -std= → do nothing
else
CXXFLAGS += -std=c++23
endif

CPPFLAGS += -I$(INCLUDE_DIR) -msse4.2 -mfma

MODULE_SRCS := $(shell find $(MODULES_DIR) -name '*.cppm') $(shell find $(SRC_DIR) -name '*.cppm') \
$(shell find $(TEST_DIR) -name '*.cppm')
TU_SRCS := $(shell find $(SRC_DIR) -name '*.cpp') $(shell find $(TEST_DIR) -name '*.cpp')
ALL_SRCS := $(MODULE_SRCS) $(TU_SRCS)

JDEP_TARGETS := $(addsuffix .jdep, $(ALL_SRCS:$(ROOT_DIR)/%=%))
ARG_TARGETS := $(addsuffix .args, $(ALL_SRCS:$(ROOT_DIR)/%=%))

BMI_TARGETS := $(addsuffix .pcm, $(MODULE_SRCS:$(ROOT_DIR)/%=%))
BMID_TARGETS := $(addsuffix .d, $(MODULE_SRCS:$(ROOT_DIR)/%=%))
BMIPP_TARGETS := $(addsuffix .i, $(MODULE_SRCS:$(ROOT_DIR)/%=%))
BMIJCMD_TARGETS := $(addsuffix .jcmd, $(MODULE_SRCS:$(ROOT_DIR)/%=%))

OBJ_TARGETS := $(addsuffix .o, $(TU_SRCS:$(ROOT_DIR)/%=%))
OBJD_TARGETS := $(addsuffix .d, $(TU_SRCS:$(ROOT_DIR)/%=%))
OBJPP_TARGETS := $(addsuffix .i, $(TU_SRCS:$(ROOT_DIR)/%=%))
OBJJCMD_TARGETS := $(addsuffix .jcmd, $(TU_SRCS:$(ROOT_DIR)/%=%))

ALL_TARGETS := $(JDEP_TARGETS) $(ARG_TARGETS) $(BMI_TARGETS) $(BMID_TARGETS) \
$(BMIPP_TARGETS) $(BMIJCMD_TARGETS) $(OBJ_TARGETS) $(OBJD_TARGETS) $(OBJPP_TARGETS) \
$(OBJJCMD_TARGETS)

.PHONY: tests clean compile.command link.command jmap jgraph FORCE print all compile_commands $(ALL_TARGETS)

.SECONDARY:  %/ $(OUTPUT_DIR)/%.d $(OUTPUT_DIR)/%.i $(OUTPUT_DIR)/%.command \
$(OUTPUT_DIR)/%.stamp  $(OUTPUT_DIR)/%.jcmd $(OUTPUT_DIR)/%.jdep $(OUTPUT_DIR)/%.args \
$(OUTPUT_DIR)/%jmap $(OUTPUT_DIR)/compile_commands.json

all: $(BMI_TARGETS) $(OUTPUT_DIR)/compile_commands.json
	@

clean:
	@rm -f $(OUTPUT_DIR)/**/*.o $(OUTPUT_DIR)/**/*.pcm  $(OUTPUT_DIR)/**/*.jcmd $(OUTPUT_DIR)/**/*.jdep \
	$(OUTPUT_DIR)/**/*.d  $(OUTPUT_DIR)/**/*.args $(OUTPUT_DIR)/**/*.jgraph $(OUTPUT_DIR)/**/*.jmap \
	$(OUTPUT_DIR)/compile.command $(OUTPUT_DIR)/link.command $(OUTPUT_DIR)/env.stamp

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

jmap: $(OUTPUT_DIR)/dpl.jmap
	@

jgraph: $(OUTPUT_DIR)/dpl.jgraph
	@

compile_commands: $(OUTPUT_DIR)/compile_commands.json
	@

$(BMI_TARGETS):%.cppm.pcm: $(OUTPUT_DIR)/%.cppm.pcm
	@

$(OBJ_TARGETS):%.cpp.o: $(OUTPUT_DIR)/%.cpp.o
	@

$(ARG_TARGETS):%.args: $(OUTPUT_DIR)/%.args
	@

$(JDEP_TARGETS):%.jdep: $(OUTPUT_DIR)/%.jdep
	@

$(OUTPUT_DIR)/dpl.jmap: $(TOOLS_DIR)/jdeps-to-jmap.jq $(addprefix $(OUTPUT_DIR)/,$(JDEP_TARGETS)) 
	@jq -s -f $^ > $@

$(OUTPUT_DIR)/dpl.jgraph: $(TOOLS_DIR)/jdeps-to-jgraph.jq $(addprefix $(OUTPUT_DIR)/,$(JDEP_TARGETS)) 
	@jq -s -f $^ > $@

$(OUTPUT_DIR)/compile_commands.json: $(addprefix $(OUTPUT_DIR)/,$(BMIJCMD_TARGETS)) $(addprefix $(OUTPUT_DIR)/,$(OBJJCMD_TARGETS)) 
	@jq -s '.' $^ > $@

$(OUTPUT_DIR)/%.cppm.args: $(OUTPUT_DIR)/dpl.jmap $(OUTPUT_DIR)/dpl.jgraph $(OUTPUT_DIR)/%.cppm.jdep
	@jq -r --slurpfile jmap $(OUTPUT_DIR)/dpl.jmap \
     --slurpfile jgraph $(OUTPUT_DIR)/dpl.jgraph \
     --arg target "$(shell jq -r '.rules[0].provides[0]["logical-name"]' $(OUTPUT_DIR)/$*.cppm.jdep)" \
     -f $(TOOLS_DIR)/jgraph-to-args.jq \
     $(OUTPUT_DIR)/$*.cppm.jdep > $@

$(OUTPUT_DIR)/%.cppm.pcm: $(ROOT_DIR)/%.cppm $(OUTPUT_DIR)/%.cppm.args
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.pcm=.d)' -MT '$@' --precompile $< -o '$@' @$(OUTPUT_DIR)/$*.cppm.args

-include $(addprefix $(OUTPUT_DIR)/,$(BMID_TARGETS))

$(OUTPUT_DIR)/%.cppm.jcmd: $(OUTPUT_DIR)/compile.command $(OUTPUT_DIR)/%.cppm.args
	@echo "{ \"directory\": \"$(OUTPUT_DIR)\", \"command\": \"$(CXX) $(CPPFLAGS) $(CXXFLAGS) --precompile $(ROOT_DIR)/$*.cppm -o $*.cppm.pcm @$*.cppm.args\", \"file\": \"$(ROOT_DIR)/$*.cppm\" }" > $@

$(OUTPUT_DIR)/%.cppm.jdep: $(ROOT_DIR)/%.cppm
	@mkdir -p '$(@D)'
	@clang-scan-deps-21 -format=p1689 -o $@ -- $(CXX) $(CPPFLAGS) $(CXXFLAGS) --precompile $< -o $(@:.jdep=.pcm)

$(OUTPUT_DIR)/%.cppm.d: $(OUTPUT_DIR)/%.cppm.jdep $(OUTPUT_DIR)/dpl.jmap $(TOOLS_DIR)/jdep-to-d.jq
	@jq -r --slurpfile jmap $(OUTPUT_DIR)/dpl.jmap -f $(TOOLS_DIR)/jdep-to-d.jq $< > $@

$(OUTPUT_DIR)/%.cpp.args: $(OUTPUT_DIR)/dpl.jmap $(OUTPUT_DIR)/dpl.jgraph $(OUTPUT_DIR)/%.cpp.jdep
	@jq -r --slurpfile jmap $(OUTPUT_DIR)/dpl.jmap \
     --slurpfile jgraph $(OUTPUT_DIR)/dpl.jgraph \
     --arg target "$(OUTPUT_DIR)/$*.cpp.o" \
     -f $(TOOLS_DIR)/jgraph-to-args.jq \
     $(OUTPUT_DIR)/$*.cpp.jdep > $@

$(OUTPUT_DIR)/%.cpp.o: $(ROOT_DIR)/%.cpp $(OUTPUT_DIR)/%.cpp.args
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.o=.d)' -MT '$@' -c $< -o '$@' @$(OUTPUT_DIR)/$*.cpp.args

-include $(addprefix $(OUTPUT_DIR)/,$(OBJD_TARGETS))

$(OUTPUT_DIR)/%.cpp.jcmd: $(OUTPUT_DIR)/compile.command $(OUTPUT_DIR)/%.cpp.args
	@echo "{ \"directory\": \"$(OUTPUT_DIR)\", \"command\": \"$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $(ROOT_DIR)/$*.cpp -o $*.cpp.pcm @$*.cpp.args\", \"file\": \"$(ROOT_DIR)/$*.cpp\" }" > $@

$(OUTPUT_DIR)/%.cpp.jdep: $(ROOT_DIR)/%.cpp
	@mkdir -p '$(@D)'
	@clang-scan-deps-21 -format=p1689 -o $@ -- $(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $(@:.jdep=.o)


TEST_OBJ_TARGETS := $(filter %.pass.cpp.o,$(OBJ_TARGETS))
PASS_OBJ_TARGETS := $(filter-out %.compile.pass.cpp.o,$(TEST_OBJ_TARGETS))
PASS_EXES := $(OUTPUT_DIR)/$(patsubst %.cpp.o,%,$(PASS_OBJ_TARGETS))
TEST_CRC := $(addsuffix .crc, $(PASS_EXES))

tests: $(TEST_CRC)
	@

$(OUTPUT_DIR)/%.pass.crc: $(OUTPUT_DIR)/%.pass
	@$< && echo "\033[0;34mTEST\033[0m $*: \033[0;32mSUCCESS\033[0m" && \
	cksum $< > $@ || { echo "TEST $(OUTPUT_DIR)/%,%,$<): \033[0;31mFAILED\033[0m" && rm -f $@; exit 1; }

$(OUTPUT_DIR)/%.pass: $(OUTPUT_DIR)/%.pass.cpp.o $(OUTPUT_DIR)/link.command
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $< $(LDLIBS) -o '$@'

