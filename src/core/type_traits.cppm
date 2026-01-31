// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.type_traits;
export import :core.fwd;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;
import :core.concepts;

// IWYU pragma: begin_exports
#include "dpl/core/type_traits/array_for.h"
#include "dpl/core/type_traits/basic_element.h"
#include "dpl/core/type_traits/basic_type.h"
#include "dpl/core/type_traits/bit_type.h"
#include "dpl/core/type_traits/common_abi.h"
#include "dpl/core/type_traits/common_arithmetic_type.h"
#include "dpl/core/type_traits/common_bits_type.h"
#include "dpl/core/type_traits/common_float_type.h"
#include "dpl/core/type_traits/common_order_type.h"
#include "dpl/core/type_traits/common_size_type.h"
#include "dpl/core/type_traits/element_count.h"
#include "dpl/core/type_traits/iota_sequence.h"
#include "dpl/core/type_traits/rebind_simd.h"
#include "dpl/core/type_traits/simd_element_type.h"
#include "dpl/core/type_traits/to_simd_mask_type.h"
#include "dpl/core/type_traits/to_simd_type.h"
// IWYU pragma: end_exports
