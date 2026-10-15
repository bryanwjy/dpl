

REQUIRED_VARS := BINARY_DIR MPFR_BIN

$(foreach v,$(REQUIRED_VARS),\
    $(if $(strip $($(v))),,$(error $(v) is not set; invoke from the parent makefile)))

$(foreach v,$(REQUIRED_VARS),\
    $(if $(filter /%,$($(v))),,$(error $(v) must be an absolute path, got '$($(v))')))

TESTDATA_DIR := $(BINARY_DIR)/testdata
JDIR_FIILES := $(shell find $(BINARY_DIR) -type f -name '*.pass.cpp.jdir' 2>/dev/null)

.PHONY: testdata

testdata.json: 
