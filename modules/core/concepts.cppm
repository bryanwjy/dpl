// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.concepts;
export import :core.fwd;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;

// IWYU pragma: begin_exports
#include "dpl/core/concepts/arithmetic_type.h"
#include "dpl/core/concepts/basic_type.h"
#include "dpl/core/concepts/broadcastable_to.h"
#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_arithmetic_with.h"
#include "dpl/core/concepts/common_basic_element_with.h"
#include "dpl/core/concepts/common_bits_with.h"
#include "dpl/core/concepts/common_class_with.h"
#include "dpl/core/concepts/common_float_with.h"
#include "dpl/core/concepts/common_integral_with.h"
#include "dpl/core/concepts/common_order_with.h"
#include "dpl/core/concepts/compatible_mask_with.h"
#include "dpl/core/concepts/immediate_like.h"
#include "dpl/core/concepts/immediate_mask_like.h"
#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_class.h"
#include "dpl/core/concepts/simd_element.h"
#include "dpl/core/concepts/simd_equivalence.h"
#include "dpl/core/concepts/simd_mask_type.h"
#include "dpl/core/concepts/simd_type.h"
// IWYU pragma: end_exports
