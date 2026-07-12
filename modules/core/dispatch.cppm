// Copyright 2026 Bryan Wong
module;

#define DPL_MODULES 1
#include "dpl/config.h"

export module dpl:core.dispatch;
export import :core.fwd;
import :core.details.dispatch;

__DPL_DEFAULT_NAMESPACE_BEGIN
// NOLINTBEGIN(misc-unused-using-decls)

namespace datapar {
inline namespace cpo {
using __DPL datapar::cpo::evaluate;
}

using __DPL datapar::maskable_simd_operation;
using __DPL datapar::same_operation_as;
using __DPL datapar::simd_algorithm_operation;
using __DPL datapar::simd_basic_operation;
using __DPL datapar::simd_canonical_invocable;
using __DPL datapar::simd_extension_invocable;
using __DPL datapar::simd_invocable;
using __DPL datapar::simd_math_operation;
using __DPL datapar::simd_operation;
using __DPL datapar::simd_primitive_operation;

} // namespace datapar

// NOLINTEND(misc-unused-using-decls)
__DPL_DEFAULT_NAMESPACE_END
