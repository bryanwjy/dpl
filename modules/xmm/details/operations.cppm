// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

#include <immintrin.h>
export module dpl.xmm:details.operations;
import :basic;
import dpl;

DPL_DISABLE_WARNING_PUSH()
#if DPL_COMPILER_MSVC
DPL_DISABLE_WARNING(5244)
#endif

// IWYU pragma: begin_exports
#include "dpl/xmm/operations/arithmetic.h"
#include "dpl/xmm/operations/bit.h"
#include "dpl/xmm/operations/bitwise.h"
#include "dpl/xmm/operations/element_cast.h"
#include "dpl/xmm/operations/reinterpret.h"
#include "dpl/xmm/operations/select.h"
#include "dpl/xmm/operations/to_simd_mask.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()
