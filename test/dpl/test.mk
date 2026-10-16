# Copyright 2025-2026 Bryan Wong

CURRENT_LIST_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
CURRENT_SOURCE_DIR := $(patsubst %/,%,$(dir $(CURRENT_LIST_PATH)))
CURRENT_BINARY_DIR := $(CURRENT_SOURCE_DIR:$(ROOT_DIR)/%=$(OUTPUT_DIR)/%)
CURRENT_DIRS  := $(sort $(call rdirs,$(CURRENT_SOURCE_DIR)))
CURRENT_SRCS  := $(sort $(call rwildcard,$(CURRENT_SOURCE_DIR),*.cppm *.cpp))
CURRENT_STEMS := $(patsubst $(CURRENT_SOURCE_DIR)/%,$(CURRENT_BINARY_DIR)/%,$(CURRENT_SRCS))
CURRENT_DEPS  := $(addsuffix .d, $(CURRENT_STEMS))
CURRENT_OBJS  := $(addsuffix .o, $(CURRENT_STEMS))
CURRENT_PCMS  := $(addsuffix .pcm, $(filter %.cppm, $(CURRENT_STEMS)))
CURRENT_JSCANS := $(addsuffix .jscan, $(CURRENT_STEMS))
CURRENT_JCMDS := $(addsuffix .jcmd, $(CURRENT_STEMS))
ALL_TARGETS += $(CURRENT_OBJS) $(CURRENT_PCMS) $(CURRENT_JSCANS) $(CURRENT_JCMDS)
ALL_TARGETS += $(CURRENT_BINARY_DIR)/env.stamp $(CURRENT_BINARY_DIR)/compile.command $(CURRENT_BINARY_DIR)/link.command
ALL_TARGETS += $(CURRENT_BINARY_DIR)/jmap.json $(CURRENT_BINARY_DIR)/jgraph.json $(CURRENT_BINARY_DIR)/module_implementations.txt $(CURRENT_BINARY_DIR)/module_dependencies.json
ALL_TARGETS += $(CURRENT_BINARY_DIR)/scan_barrier.mk $(CURRENT_BINARY_DIR)/scan_commands.json
ALL_TARGETS += $(CURRENT_OBJS:.o=.module.rsp)
ALL_SOURCES += $(CURRENT_SRCS)

COMPILE_PASS_TARGETS := $(filter %.compile.pass.cpp.o, $(CURRENT_OBJS))
PASS_TARGETS := $(filter-out %.compile.pass,$(patsubst %.cpp.o, %,$(filter %.pass.cpp.o,$(CURRENT_OBJS))))
TEST_TARGETS := $(COMPILE_PASS_TARGETS) $(PASS_TARGETS)
ALL_TARGETS += $(TEST_TARGETS)
ALL_TARGETS += $(addsuffix .cpp.link.rsp, $(PASS_TARGETS))

COMPILE_PASS_CRC := $(COMPILE_PASS_TARGETS:.cpp.o=.crc)
PASS_CRC := $(addsuffix .crc,$(PASS_TARGETS))
TEST_CRC := $(PASS_CRC) $(COMPILE_PASS_CRC)
ALL_TARGETS += $(TEST_CRC)
TEST_ALIAS := $(TEST_CRC:$(OUTPUT_DIR)/%.crc=%)
MATHDATA_DIR := $(CURRENT_SOURCE_DIR)/mathdata

define subdir_to_crc
$(patsubst $(ROOT_DIR)/%.cpp,$(OUTPUT_DIR)/%.crc,$(wildcard $(ROOT_DIR)/$(1)/*.pass.cpp)) \
$(patsubst $(ROOT_DIR)/%.cpp,$(OUTPUT_DIR)/%.crc,$(wildcard $(ROOT_DIR)/$(1)/**/*.pass.cpp))
endef

TEST_SUBDIRS := $(CURRENT_DIRS:$(ROOT_DIR)/%=%)


$(TEST_ALIAS):%.pass: $(OUTPUT_DIR)/%.pass.crc
$(TEST_SUBDIRS):%: $$(call subdir_to_crc,%)
.PHONY: $(TEST_SUBDIRS) $(TEST_ALIAS) parallel_probes

$(CLEAN_MANIFEST): ALL_TARGETS += $(CURRENT_DEPS)

$(CURRENT_BINARY_DIR)/%: private CPPFLAGS += -I$(MATHDATA_DIR)
$(CURRENT_BINARY_DIR)/%: private CURRENT_BINARY_DIR := $(CURRENT_BINARY_DIR)
$(CURRENT_BINARY_DIR)/%: private CURRENT_SOURCE_DIR := $(CURRENT_SOURCE_DIR)
$(CURRENT_BINARY_DIR)/%: private CURRENT_OBJS := $(CURRENT_OBJS)
$(CURRENT_BINARY_DIR)/%: private CURRENT_LIST_PATH := $(CURRENT_LIST_PATH)
$(CURRENT_BINARY_DIR)/%.pass.cpp.o $(CURRENT_BINARY_DIR)/%.pass.cpp.jcmd: private COMPILE_RESPONSES += tflags

$(CURRENT_BINARY_DIR)/%.crc: $(CURRENT_BINARY_DIR)/%
	@$< && echo "\033[0;34mTEST\033[0m $*: \033[0;32mSUCCESS\033[0m" && \
	cksum $< > $@ || { echo "TEST $*): \033[0;31mFAILED\033[0m" && rm -f $@; exit 1; }

$(CURRENT_BINARY_DIR)/%.compile.pass.crc: $(CURRENT_BINARY_DIR)/%.compile.pass.cpp.o
	@echo "\033[0;34mCOMPILE\033[0m $*.compile.pass: \033[0;32mSUCCESS\033[0m" && \
	cksum $< > $@ || { echo "\033[0;34mCOMPILE\033[0m $*.compile.pass: \033[0;31mFAILED\033[0m"; exit 1; }

$(PASS_TARGETS):$(CURRENT_BINARY_DIR)/%.pass: $(CURRENT_BINARY_DIR)/%.pass.cpp.o \
	$(CURRENT_BINARY_DIR)/link.command \
	$$(foreach p,$$(LINK_RESPONSES),$(CURRENT_BINARY_DIR)/%.pass.cpp.$$p.rsp)
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o '$@' $< $(LDLIBS) \
	$(foreach p,$(LINK_RESPONSES),@$(CURRENT_BINARY_DIR)/$*.pass.cpp.$p.rsp)

$(CURRENT_BINARY_DIR)/env.stamp: FORCE
	@$(call replace_if_different, printf "%s\n" \
	  "CXX=$(CXX)" \
	  "CPPFLAGS=$(CPPFLAGS)" \
	  "CXXFLAGS=$(CXXFLAGS)" \
	  "LDFLAGS=$(LDFLAGS)" \
	  "LDLIBS=$(LDLIBS)")

$(CURRENT_BINARY_DIR)/compile.command: $(CURRENT_LIST_PATH) $(CURRENT_BINARY_DIR)/env.stamp
	@$(call replace_if_different, echo "$(CXX) $(CPPFLAGS) $(CXXFLAGS)")
$(CURRENT_BINARY_DIR)/link.command: $(CURRENT_LIST_PATH) $(CURRENT_BINARY_DIR)/env.stamp
	@$(call replace_if_different, echo "$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) $(LDLIBS)")

$(CURRENT_BINARY_DIR)/%.cppm.o: $(CURRENT_BINARY_DIR)/%.cppm.pcm;
$(CURRENT_BINARY_DIR)/%.cppm.pcm: $(CURRENT_SOURCE_DIR)/%.cppm \
	$$(foreach p,$$(COMPILE_RESPONSES) $$(PREPROCESSOR_RESPONSES),$(CURRENT_BINARY_DIR)/%.cppm.$$p.rsp) \
	$(CURRENT_BINARY_DIR)/compile.command | $(CURRENT_BINARY_DIR)/scan_barrier.mk
	@set -e;\
	tmppcm=$$(mktemp); tmpobj=$$(mktemp);\
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.pcm=.d)' -MT '$@' \
	  -fmodule-output="$$tmppcm" -fmodules-reduced-bmi -c $< -o "$$tmpobj" \
	  $(foreach p,$(PREPROCESSOR_RESPONSES) $(COMPILE_RESPONSES),@$(@:.pcm=.$p.rsp)); \
	if cmp -s "$$tmppcm" '$@' 2>/dev/null; then \
	    rm "$$tmppcm"; \
	else \
	    mv "$$tmppcm" '$@'; \
	fi; \
	if cmp -s "$$tmpobj" '$(@:.pcm=.o)' 2>/dev/null; then \
	    rm "$$tmpobj"; \
	else \
	    mv "$$tmpobj" '$(@:.pcm=.o)'; \
	fi

$(CURRENT_BINARY_DIR)/%.cpp.o: $(CURRENT_SOURCE_DIR)/%.cpp \
	$$(foreach p,$$(COMPILE_RESPONSES) $$(PREPROCESSOR_RESPONSES),$(CURRENT_BINARY_DIR)/%.cpp.$$p.rsp) \
	$(CURRENT_BINARY_DIR)/compile.command | $(CURRENT_BINARY_DIR)/scan_barrier.mk
	@mkdir -p '$(@D)'
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MMD -MP -MF '$(@:.o=.d)' -MT '$@' -c $< -o '$@' \
	$(foreach p,$(PREPROCESSOR_RESPONSES) $(COMPILE_RESPONSES),@$(@:.o=.$p.rsp))

$(call safe_include, $(CURRENT_DEPS))

$(CURRENT_BINARY_DIR)/%.jcmd: $(TOOLS_DIR)/generate-jcmd.jq \
	$$(foreach p,$$(COMPILE_RESPONSES) $$(PREPROCESSOR_RESPONSES),$(CURRENT_BINARY_DIR)/%.$$p.rsp) \
	$(CURRENT_BINARY_DIR)/compile.command
	@$(call replace_if_different, jq -f $(TOOLS_DIR)/generate-jcmd.jq -n \
	--arg directory '$(CURRENT_BINARY_DIR)' \
	$(foreach p,$(COMPILE_RESPONSES) $(PREPROCESSOR_RESPONSES),--rawfile $p '$(@:.jcmd=.$p.rsp)') \
	--arg command '$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(if $(filter %.cppm,$*), -x c++ -fmodules-reduced-bmi )-c $(CURRENT_SOURCE_DIR)/$* -o $*.o' \
	--arg file '$(CURRENT_SOURCE_DIR)/$*' \
	--arg output '$(if $(filter %.cppm,$*),$(@:.jscan=.pcm),$(@:.jscan=.o))')

$(CURRENT_BINARY_DIR)/%.jscan: $(TOOLS_DIR)/generate-jcmd.jq \
	$$(foreach p,$$(PREPROCESSOR_RESPONSES),$(CURRENT_BINARY_DIR)/%.$$p.rsp) \
	$(CURRENT_BINARY_DIR)/compile.command
	@$(call replace_if_different, jq -f $(TOOLS_DIR)/generate-jcmd.jq -n \
	--arg directory '$(CURRENT_BINARY_DIR)' \
	$(foreach p,$(PREPROCESSOR_RESPONSES),--rawfile $p '$(@:.jscan=.$p.rsp)') \
	--arg command '$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(if $(filter %.cppm,$*),-x c++ -fmodules-reduced-bmi )-c $(CURRENT_SOURCE_DIR)/$* -o $*.o' \
	--arg file '$(CURRENT_SOURCE_DIR)/$*' \
	--arg output '$(if $(filter %.cppm,$*),$(@:.jscan=.pcm),$(@:.jscan=.o))')

$(CURRENT_BINARY_DIR)/%.module.rsp: $(CURRENT_BINARY_DIR)/jgraph.json \
	$(CURRENT_BINARY_DIR)/jmap.json $(TOOLS_DIR)/module-response.jq
	@$(call replace_if_different, $(TOOLS_DIR)/module-response.jq \
	--arg module '$(if $(filter %.cppm,$*),$(@:.module.rsp=.pcm),$(@:.module.rsp=.o))' \
	--slurpfile jmap $(CURRENT_BINARY_DIR)/jmap.json \
	-r $(CURRENT_BINARY_DIR)/jgraph.json)

$(CURRENT_BINARY_DIR)/%.link.rsp: $(TOOLS_DIR)/link-response.jq \
	$(CURRENT_BINARY_DIR)/jgraph.json $(CURRENT_BINARY_DIR)/jmap.json \
	$(CURRENT_BINARY_DIR)/module_implementations.txt
	@$(call replace_if_different, $(TOOLS_DIR)/link-response.jq \
	--arg module '$(@:.link.rsp=.o)' \
	--slurpfile jmap $(CURRENT_BINARY_DIR)/jmap.json \
	--rawfile impl $(CURRENT_BINARY_DIR)/module_implementations.txt \
    -r $(CURRENT_BINARY_DIR)/jgraph.json)

$(CURRENT_BINARY_DIR)/scan_commands.json: $(CURRENT_JSCANS) $(CURRENT_DIRS) | $(CURRENT_BINARY_DIR)/
	@$(call replace_if_different, jq -s '.' $(filter %.jscan,$^))

$(CURRENT_BINARY_DIR)/module_dependencies.json: $(CURRENT_BINARY_DIR)/scan_commands.json $(CURRENT_SRCS) | $(CURRENT_BINARY_DIR)/
	@$(SCAN_DEPS) -format=p1689 -compilation-database=$< -o $@.tmp
	@cmp -s $@.tmp $@ 2>/dev/null && rm $@.tmp || mv $@.tmp $@

$(CURRENT_BINARY_DIR)/scan_barrier.mk: $(TOOLS_DIR)/jdep-to-d.jq
$(CURRENT_BINARY_DIR)/scan_barrier.mk: $(CURRENT_BINARY_DIR)/jmap.json $(CURRENT_BINARY_DIR)/module_dependencies.json
	@jq -f $(TOOLS_DIR)/jdep-to-d.jq --slurpfile jmap $(CURRENT_BINARY_DIR)/jmap.json -r $(CURRENT_BINARY_DIR)/module_dependencies.json > $(CURRENT_BINARY_DIR)/scan_barrier.mk

$(call safe_include,$(CURRENT_BINARY_DIR)/scan_barrier.mk)

$(CURRENT_BINARY_DIR)/module_implementations.txt: $(CURRENT_DIRS)
	@mkdir -p '$(@D)'
	@$(file >$@.tmp,$(subst $(space),$(newline),$(CURRENT_OBJS)))
	@cmp -s $@.tmp $@ 2>/dev/null && rm $@.tmp || mv $@.tmp $@

$(CURRENT_BINARY_DIR)/jmap.json: $(OUTPUT_DIR)/test/support/jmap.json
$(CURRENT_BINARY_DIR)/jmap.json: $(TOOLS_DIR)/jdeps-to-jmap.jq
$(CURRENT_BINARY_DIR)/jmap.json: $(CURRENT_BINARY_DIR)/module_dependencies.json
	@jq -f $(TOOLS_DIR)/jdeps-to-jmap.jq --slurpfile base $(OUTPUT_DIR)/test/support/jmap.json -s $< > $@

$(CURRENT_BINARY_DIR)/jgraph.json: $(OUTPUT_DIR)/test/support/jgraph.json
$(CURRENT_BINARY_DIR)/jgraph.json: $(TOOLS_DIR)/graph.jq 
$(CURRENT_BINARY_DIR)/jgraph.json: $(CURRENT_BINARY_DIR)/module_dependencies.json
	@jq -f $(TOOLS_DIR)/graph.jq --slurpfile base $(OUTPUT_DIR)/test/support/jgraph.json $< > $@


DIRECTIVES := $(patsubst %.o,%.jdir,$(filter %.pass.cpp.o, $(CURRENT_OBJS)))
$(CURRENT_BINARY_DIR)/%.jdir: $(CURRENT_SOURCE_DIR)/% $(TOOLS_DIR)/directives.awk
	@$(call replace_if_different, $(TOOLS_DIR)/directives.awk -v family=$(CXX_FAMILY) $<)

ALL_TARGETS += $(DIRECTIVES)

MPFR_BIN := $(OUTPUT_DIR)/test/mpfr/mpfr
ALL_TARGETS += $(OUTPUT_DIR)/mathdata.mk $(OUTPUT_DIR)/mathdata.json
.PHONY: mathdata
mathdata: $(MPFR_BIN) $(DIRECTIVES) $(TOOLS_DIR)/mathdata.jq
	@$(file >$(OUTPUT_DIR)/mathdata.json,$(foreach f,$(DIRECTIVES),{"file":"$f","directives":$(file <$f)}$(newline)))
	@jq -r -s -f $(TOOLS_DIR)/mathdata.jq --arg dir '$(MATHDATA_DIR)' --arg bin '$$(MPFR_BIN)' \
	    $(OUTPUT_DIR)/mathdata.json > $(OUTPUT_DIR)/mathdata.mk
	@$(MAKE) --no-print-directory -f $(OUTPUT_DIR)/mathdata.mk -C $(MATHDATA_DIR) MPFR_BIN='$(MPFR_BIN)' all

TFLAGS_RSPS := $(patsubst %.o,%.tflags.rsp, $(filter %.pass.cpp.o,$(CURRENT_OBJS)))
$(CURRENT_BINARY_DIR)/%.tflags.rsp: $(CURRENT_BINARY_DIR)/%.jdir $(CURRENT_BINARY_DIR)/supported_flags.txt
	@$(call replace_if_different, (jq -r '.[] | .["compile-flags"] // [] | .[]' $< | grep -Fxf $(CURRENT_BINARY_DIR)/supported_flags.txt - || true))

ALL_TARGETS += $(TFLAGS_RSPS) $(CURRENT_BINARY_DIR)/candidate_flags.txt

$(CURRENT_BINARY_DIR)/candidate_flags.txt: $(DIRECTIVES)
	@jq -r '.[] | .["compile-flags"] // [] | .[]' $^ | sort -u > $@

$(CURRENT_BINARY_DIR)/supported_flags.txt: $(CURRENT_BINARY_DIR)/candidate_flags.txt
	@mkdir -p $(CURRENT_BINARY_DIR)/.probe && \
	awk '{print $$0 > ("$(CURRENT_BINARY_DIR)/.probe/" NR ".ptrn")}' $< && \
	$(MAKE) --no-print-directory parallel_probes && cp $(CURRENT_BINARY_DIR)/.probe/supported_flags.txt $(CURRENT_BINARY_DIR)/supported_flags.txt && \
	echo "Supported DPL directive compile flags written to $(CURRENT_BINARY_DIR)/supported_flags.txt"

ifeq ($(MAKECMDGOALS),parallel_probes)
# Generate a list of 'valid' targets based on the 'ptrn' files we just created
PTRNS  := $(wildcard $(CURRENT_BINARY_DIR)/.probe/*.ptrn)
VALIDS := $(patsubst %.ptrn,%.valid,$(PTRNS))

parallel_probes: $(CURRENT_BINARY_DIR)/.probe/supported_flags.txt
	@echo "Probing DPL directive compile flags..."

$(CURRENT_BINARY_DIR)/.probe/supported_flags.txt: $(VALIDS)
	@cat $(CURRENT_BINARY_DIR)/.probe/*.valid > $@ 2>/dev/null || true

$(CURRENT_BINARY_DIR)/.probe/%.valid: $(CURRENT_BINARY_DIR)/.probe/%.ptrn
	@if echo "int main(){}" | $(CXX) -fsyntax-only @$< -x c++ - -o /dev/null >/dev/null 2>&1; then \
		cp $< $@; \
	else \
		touch $@; \
	fi
endif

$(CLEAN_MANIFEST): ALL_TARGETS += $(wildcard $(OUTPUT_DIR)/test/dpl/.probe/*.valid) $(wildcard $(OUTPUT_DIR)/test/dpl/.probe/*.ptrn) $(OUTPUT_DIR)/test/dpl/.probe/supported_flags.txt
