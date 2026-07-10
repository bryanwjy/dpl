// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.details.type_traits;
export import :core.fwd;
export import :core.type_interface;
import :std.concepts;
import :std.bit;
import :std.type_traits;
import :std.utility;

// IWYU pragma: begin_exports
#include "dpl/core/type_traits/details/fwd.h"

#include "dpl/core/type_traits/details/cpo_result.h"
#include "dpl/core/type_traits/details/declarg.h"
#include "dpl/core/type_traits/details/has_expression_result.h"
#include "dpl/core/type_traits/details/has_simd_members.h"
#include "dpl/core/type_traits/details/none_abi_type.h"
#include "dpl/core/type_traits/details/simd_abi_size.h"
// IWYU pragma: end_exports
