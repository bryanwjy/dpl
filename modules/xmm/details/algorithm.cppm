// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

#if DPL_COMPILER_CLANG
#  define static
#endif
#include <immintrin.h>
#if DPL_COMPILER_CLANG
#  undef static
#endif

export module dpl.xmm:details.algorithm;
import :basic;
import :operations;
import dpl;

DPL_DISABLE_WARNING_PUSH()
#if DPL_COMPILER_MSVC
DPL_DISABLE_WARNING(5244)
#endif

// IWYU pragma: begin_exports
#include "dpl/xmm/algorithm/lookup.h"
#include "dpl/xmm/algorithm/sad.h"
#include "dpl/xmm/algorithm/shift.h"
#include "dpl/xmm/algorithm/slide.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()
