// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

// TODO figure out how to work around clang UB

// All functions in the intel intrinsic API has been made 'static'
// But it is UB to transitively expose internal linkage function
// usage through the module interface
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
#include "dpl/xmm/math/fmaddsub.h"
#include "dpl/xmm/math/fmsub.h"
#include "dpl/xmm/math/fmsubadd.h"
#include "dpl/xmm/math/fnmadd.h"
#include "dpl/xmm/math/fnmsub.h"
// IWYU pragma: end_exports

DPL_DISABLE_WARNING_POP()
