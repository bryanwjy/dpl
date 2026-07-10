// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.type_interface;
export import :core.fwd;
import :std.concepts;
import :std.type_traits;

// IWYU pragma: begin_exports
#include "dpl/core/type_interface/enable_const_mask.h"
#include "dpl/core/type_interface/enable_simd_abi.h"
#include "dpl/core/type_interface/enable_simd_mask.h"
#include "dpl/core/type_interface/enable_simd_vector.h"
#include "dpl/core/type_interface/simd_base.h"
// IWYU pragma: end_exports
