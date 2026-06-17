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

// IWYU pragma: begin_exports
#include "dpl/core/type_traits/canonical_type.h"
#include "dpl/core/type_traits/common_abi.h"
#include "dpl/core/type_traits/common_size_type.h"
#include "dpl/core/type_traits/cpo_result.h"
#include "dpl/core/type_traits/declarg.h"
#include "dpl/core/type_traits/enable_const_mask.h"
#include "dpl/core/type_traits/enable_simd_abi.h"
#include "dpl/core/type_traits/enable_simd_mask.h"
#include "dpl/core/type_traits/enable_simd_vector.h"
#include "dpl/core/type_traits/is_canonical_type.h"
#include "dpl/core/type_traits/rebind_simd.h"
#include "dpl/core/type_traits/representation.h"
#include "dpl/core/type_traits/simd_abi_traits.h"
#include "dpl/core/type_traits/simd_abi_type.h"
#include "dpl/core/type_traits/simd_element_representation.h"
#include "dpl/core/type_traits/simd_element_type.h"
#include "dpl/core/type_traits/simd_expression_result.h"
#include "dpl/core/type_traits/simd_traits.h"
#include "dpl/core/type_traits/simd_vector_type.h"
// IWYU pragma: end_exports
