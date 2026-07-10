// Copyright 2025 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.type_traits;
export import :core.fwd;
export import :core.type_interface;
import :core.details.type_traits;
import :core.numbers;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;

// IWYU pragma: begin_exports
#include "dpl/core/type_traits/canonical_type.h"
#include "dpl/core/type_traits/common_abi.h"
#include "dpl/core/type_traits/common_size_type.h"
#include "dpl/core/type_traits/rebind_simd.h"
#include "dpl/core/type_traits/representation.h"
#include "dpl/core/type_traits/simd_abi_traits.h"
#include "dpl/core/type_traits/simd_abi_type.h"
#include "dpl/core/type_traits/simd_element_representation.h"
#include "dpl/core/type_traits/simd_element_type.h"
#include "dpl/core/type_traits/simd_expression_result.h"
#include "dpl/core/type_traits/simd_mask_type.h"
#include "dpl/core/type_traits/simd_native_type.h"
#include "dpl/core/type_traits/simd_value_type.h"
#include "dpl/core/type_traits/simd_vector_type.h"
// IWYU pragma: end_exports
