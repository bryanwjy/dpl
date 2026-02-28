// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

#include <immintrin.h>

export module dpl.xmm:operations;
export import dpl;
import :basic;

// IWYU pragma: begin_exports
#include "dpl/xmm/operations/abs.h"
#include "dpl/xmm/operations/arithmetic.h"
#include "dpl/xmm/operations/bit.h"
// IWYU pragma: end_exports
