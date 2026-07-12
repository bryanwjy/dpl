// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.fwd;
export import :std.stddef;

DPL_DISABLE_WARNING_PUSH()
#if DPL_COMPILER_MSVC
DPL_DISABLE_WARNING(5244)
#endif

// IWYU pragma: begin_exports
#include "dpl/core/fwd/basic.h"
#include "dpl/core/fwd/namespace.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()
