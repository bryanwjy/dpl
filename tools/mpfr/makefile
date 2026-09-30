# Copyright 2026 Bryan Wong

MKFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR := $(patsubst %/,%,$(dir $(MKFILE_PATH)))

REQUIRED_VARS := BINARY_DIR DPL_INCLUDE_DIR DPL_JMAP DPL_JGRAPH

$(foreach v,$(REQUIRED_VARS),\
    $(if $(strip $($(v))),,$(error $(v) is not set; invoke from the parent makefile)))

$(foreach v,$(REQUIRED_VARS),\
    $(if $(filter /%,$($(v))),,$(error $(v) must be an absolute path, got '$($(v))')))


ROOT_DIR := $(MKFILE_DIR)
OUTPUT_DIR := $(MKFILE_DIR)/build
TOOLS_DIR := $(MKFILE_DIR)/..
CXX := clang++-23
SCAN_DEPS := clang-scan-deps-21

# Hard-code for now
CXX_FAMILY := clang

ifeq ($(filter -std=%,$(CXXFLAGS)),)
CXXFLAGS += -std=c++23
endif


CPPFLAGS += -I$(DPL_INCLUDE_DIR) -DMPFR_USE_NO_MACRO
CXXFLAGS += -O2 -march=native -fvisibility=hidden -fvisibility-inlines-hidden -ftime-trace
LDFLAGS += -lmpfr -lgmp

ALL_SOURCES := $(shell find $(MKFILE_DIR) -type f \( -name '*.cpp' -o -name '*.cppm' \) 2>/dev/null)
CPP_SOURCES := $(filter %.cpp,$(ALL_SOURCES))
MODULE_SOURCES := $(filter %.cppm,$(ALL_SOURCES))
EXT_JMAP := $(DPL_JMAP)
EXT_JGRAPH := $(DPL_JGRAPH)

# Precompiled modules
PCM_TARGETS := $(addsuffix .pcm, $(MODULE_SOURCES:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
# Object files
OBJ_TARGETS := $(addsuffix .o, $(ALL_SOURCES:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))
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
# Directive files (Extra compile flags defined in the test file)
JDIR_TARGETS := $(addsuffix .jdir, $(OUTPUT_STEMS))
# Link response files
LRSP_TARGETS := $(addsuffix .lrsp, $(OUTPUT_STEMS))
OUTPUT_STEMS :=

DEP_FILES := $(addsuffix .d, $(ALL_SOURCES:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%))

ALL_TARGETS := $(PCM_TARGETS) $(MRSP_TARGETS) $(JCMD_TARGETS) $(JSCAN_TARGETS) \
$(OBJ_TARGETS) $(LRSP_TARGETS) $(MOBJ_TARGETS) $(JDIR_TARGETS) $(PASS_TARGETS)

BUILD_JSON := $(OUTPUT_DIR)/jmap.json $(OUTPUT_DIR)/jgraph.json $(OUTPUT_DIR)/compile_commands.json \
$(OUTPUT_DIR)/module_dependencies.json $(OUTPUT_DIR)/scan_commands.json
BUILD_TXT := $(OUTPUT_DIR)/candidate_flags.txt $(OUTPUT_DIR)/module_implementations.txt
BUILD_FILES := $(BUILD_JSON) $(BUILD_TXT)
BUILD_JSON_ALIAS := $(basename $(notdir $(BUILD_JSON)))
BUILD_TXT_ALIAS := $(basename $(notdir $(BUILD_TXT)))
MODULE_ALIAS := $(basename $(MODULE_SOURCES:$(ROOT_DIR)/%=%))
BUILD_MANIFEST := $(OUTPUT_DIR)/build_manifest.txt

.PHONY: all clean FORCE $(BUILD_JSON_ALIAS) $(BUILD_TXT_ALIAS) $(MODULE_ALIAS)

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

empty :=
space := $(empty) $(empty)
comma := ,
# newline requiures two empty lines so that dereferencing it results in a single newline
define newline


endef

all: $(BINARY_DIR)/mpfr $(BINARY_DIR)/interface.cppm.pcm

clean:
	@if [ -s $(BUILD_MANIFEST) ]; then \
	    xargs rm -f < $(BUILD_MANIFEST) && rm $(BUILD_MANIFEST); \
	else \
		rm -f $(TEST_CRC) $(ALL_TARGETS) $(BUILD_FILES) $(DEP_FILES) $(OUTPUT_DIR)/scan_barrier.d $(PROBE_FILES); \
	fi

ifneq ($(MAKECMDGOALS),clean)

$(OUTPUT_DIR)/build_manifest.txt: $(ALL_SOURCES)
	@$(file >$@,$(foreach f,$(sort $(ALL_TARGETS) $(BUILD_FILES) $(DEP_FILES) $(OUTPUT_DIR)/scan_barrier.d),$(f)$(newline)))

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

$(OUTPUT_DIR)/jmap.json: $(TOOLS_DIR)/jdeps-to-jmap.jq $(OUTPUT_DIR)/module_dependencies.json
	@jq -f $(TOOLS_DIR)/jdeps-to-jmap.jq $(if $(EXT_JMAP),--slurpfile base '$(EXT_JMAP)') -s $(OUTPUT_DIR)/module_dependencies.json > $@

$(OUTPUT_DIR)/jgraph.json: $(TOOLS_DIR)/graph.jq $(OUTPUT_DIR)/module_dependencies.json
	@jq -f $(TOOLS_DIR)/graph.jq $(if $(EXT_JGRAPH),--slurpfile base '$(EXT_JGRAPH)') $(OUTPUT_DIR)/module_dependencies.json > $@

$(OUTPUT_DIR)/compile_commands.json: $(JCMD_TARGETS)
	@jq -s '.' $^ > $@

$(OUTPUT_DIR)/scan_commands.json: $(JSCAN_TARGETS)
	@jq -s '.' $^ > $@

$(OUTPUT_DIR)/module_dependencies.json: $(OUTPUT_DIR)/scan_commands.json $(ALL_SOURCES)
	@mkdir -p '$(@D)'
	@$(SCAN_DEPS) -format=p1689 -compilation-database=$< -o $@.tmp
	@cmp -s $@.tmp $@ 2>/dev/null && rm $@.tmp || mv $@.tmp $@

$(OUTPUT_DIR)/module_implementations.txt: FORCE
	@mkdir -p '$(@D)'
	@$(file >$@.tmp,$(subst $(space),$(newline),$(sort $(MOBJ_TARGETS))))
	@cmp -s $@.tmp $@ 2>/dev/null && rm $@.tmp || mv $@.tmp $@

$(OUTPUT_DIR)/scan_barrier.d: $(OUTPUT_DIR)/jmap.json $(OUTPUT_DIR)/module_dependencies.json $(TOOLS_DIR)/jdep-to-d.jq
	@jq -f $(TOOLS_DIR)/jdep-to-d.jq \
	--argjson whitelist '$(or $(JMAP_WHITELIST),[])' \
	--slurpfile jmap $(OUTPUT_DIR)/jmap.json \
	-r $(OUTPUT_DIR)/module_dependencies.json > $(OUTPUT_DIR)/scan_barrier.d

-include $(OUTPUT_DIR)/scan_barrier.d

$(MODULE_ALIAS):%: $(OUTPUT_DIR)/%.cppm.pcm
$(BUILD_JSON_ALIAS):%: $(OUTPUT_DIR)/%.json
$(BUILD_TXT_ALIAS):%: $(OUTPUT_DIR)/%.txt
.SECONDEXPANSION:
$(MOBJ_TARGETS):%.o: %.pcm

mpfr: $(BINARY_DIR)/mpfr $(BINARY_DIR)/interface.cppm.pcm
	@

$(BINARY_DIR)/mpfr: $(OUTPUT_DIR)/mpfr
	@cp $< $@

$(BINARY_DIR)/interface.cppm.pcm: $(OUTPUT_DIR)/interface.cppm.pcm
	@cp $< $@

$(OUTPUT_DIR)/mpfr: $(OBJ_TARGETS) $(MOBJ_TARGETS)
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS) -o '$@' $^

$(PCM_TARGETS):$(OUTPUT_DIR)/%.cppm.pcm: $(ROOT_DIR)/%.cppm $(OUTPUT_DIR)/%.cppm.mrsp $(OUTPUT_DIR)/compile.command | $(OUTPUT_DIR)/scan_barrier.d
	@tmpfile=$$(mktemp); \
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.pcm=.d)' -MT '$@' \
	  -fmodule-output="$$tmpfile" -fmodules-reduced-bmi -c $< -o '$(@:.pcm=.o)' @$(@:.pcm=.mrsp); \
	if cmp -s "$$tmpfile" '$@' 2>/dev/null; then \
	    rm "$$tmpfile"; \
	else \
	    mv "$$tmpfile" '$@'; \
	fi

$(OBJ_TARGETS):$(OUTPUT_DIR)/%.cpp.o: $(ROOT_DIR)/%.cpp $(OUTPUT_DIR)/%.cpp.mrsp $(OUTPUT_DIR)/compile.command | $(OUTPUT_DIR)/scan_barrier.d
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
	$(call replace_if_different, $(TOOLS_DIR)/link-response.jq \
	--arg module '$*.o' \
	--slurpfile jmap $(OUTPUT_DIR)/jmap.json \
	--rawfile impl $(OUTPUT_DIR)/module_implementations.txt \
    -r $(OUTPUT_DIR)/jgraph.json)

endif

