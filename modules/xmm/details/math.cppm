// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

#include <immintrin.h>
export module dpl.xmm:details.math;
import :basic;
import :operations;
import dpl;

DPL_DISABLE_WARNING_PUSH()
#if DPL_COMPILER_MSVC
DPL_DISABLE_WARNING(5244)
#endif

// IWYU pragma: begin_exports
#include "dpl/xmm/math/fmadd.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()
