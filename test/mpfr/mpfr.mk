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
ALL_TARGETS += $(CURRENT_BINARY_DIR)/mpfr
ALL_TARGETS += $(CURRENT_BINARY_DIR)/mpfr.cpp.link.rsp
ALL_TARGETS += $(CURRENT_OBJS:.o=.module.rsp)
ALL_SOURCES += $(CURRENT_SRCS)

$(CLEAN_MANIFEST): ALL_TARGETS += $(CURRENT_DEPS)

$(CURRENT_BINARY_DIR):
	@mkdir -p $@

$(CURRENT_BINARY_DIR)/%: private CURRENT_BINARY_DIR := $(CURRENT_BINARY_DIR)
$(CURRENT_BINARY_DIR)/%: private CURRENT_SOURCE_DIR := $(CURRENT_SOURCE_DIR)
$(CURRENT_BINARY_DIR)/%: private CURRENT_OBJS := $(CURRENT_OBJS)
$(CURRENT_BINARY_DIR)/%: private CPPFLAGS += -DMPFR_USE_NO_MACRO
$(CURRENT_BINARY_DIR)/%: private LDFLAGS += -lmpfr -lgmp
$(CURRENT_BINARY_DIR)/%: private CXXFLAGS += -ftime-trace -O2

$(CURRENT_BINARY_DIR)/mpfr: $(CURRENT_BINARY_DIR)/mpfr.cpp.o \
	$$(foreach p,$$(LINK_RESPONSES),$(CURRENT_BINARY_DIR)/mpfr.cpp.$$p.rsp) \
	$(CURRENT_BINARY_DIR)/link.command | $(CURRENT_BINARY_DIR)/scan_barrier.mk
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(LDFLAGS) -o '$@' $< $(LDLIBS) \
	$(foreach p,$(LINK_RESPONSES),@$@.cpp.$p.rsp)


$(CURRENT_BINARY_DIR)/env.stamp: FORCE | $(CURRENT_BINARY_DIR)
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
	@$(call replace_if_different, jq -f $< -n \
	--arg directory '$(CURRENT_BINARY_DIR)' \
	$(foreach p,$(PREPROCESSOR_RESPONSES) $(COMPILE_RESPONSES),--rawfile $p '$(@:.jcmd=.$p.rsp)') \
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

$(CURRENT_BINARY_DIR)/%.module.rsp: $(CURRENT_BINARY_DIR)/jgraph.json $(CURRENT_BINARY_DIR)/jmap.json $(TOOLS_DIR)/module-response.jq
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

$(CURRENT_BINARY_DIR)/scan_commands.json: $(CURRENT_JSCANS) $(CURRENT_DIRS)
	@$(call replace_if_different, jq -s '.' $(filter %.jscan,$^))

$(CURRENT_BINARY_DIR)/module_dependencies.json: $(CURRENT_BINARY_DIR)/scan_commands.json $(CURRENT_SRCS) | $(CURRENT_BINARY_DIR)/
	@$(SCAN_DEPS) -format=p1689 -compilation-database=$< -o $@.tmp
	@cmp -s $@.tmp $@ 2>/dev/null && rm $@.tmp || mv $@.tmp $@

$(CURRENT_BINARY_DIR)/scan_barrier.mk: $(TOOLS_DIR)/jdep-to-d.jq
$(CURRENT_BINARY_DIR)/scan_barrier.mk: $(CURRENT_BINARY_DIR)/jmap.json $(CURRENT_BINARY_DIR)/module_dependencies.json
	@jq -f $(TOOLS_DIR)/jdep-to-d.jq --slurpfile jmap $(CURRENT_BINARY_DIR)/jmap.json -r $(CURRENT_BINARY_DIR)/module_dependencies.json > $(CURRENT_BINARY_DIR)/scan_barrier.mk

$(call safe_include, $(CURRENT_BINARY_DIR)/scan_barrier.mk)

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
