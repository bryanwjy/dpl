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
import :core.type_traits;

// IWYU pragma: begin_exports
#include "dpl/core/concepts/broadcastable_to.h"
#include "dpl/core/concepts/canonical.h"
#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_simd_type_with.h"
#include "dpl/core/concepts/common_size_with.h"
#include "dpl/core/concepts/decayable.h"
#include "dpl/core/concepts/equivalence.h"
#include "dpl/core/concepts/extended.h"
#include "dpl/core/concepts/mask_compatibility.h"
#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_element.h"
#include "dpl/core/concepts/simd_expression.h"
#include "dpl/core/concepts/simd_mask.h"
#include "dpl/core/concepts/simd_type.h"
#include "dpl/core/concepts/simd_vector.h"
// IWYU pragma: end_exports
