
// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_expression.h" // IWYU pragma: export
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
struct simd_expression_result {
    using type DPL_NODEBUG = internal::expression_result_t<T>;
};
DPL_EXPORT template <typename T>
using simd_expression_result_t = internal::expression_result_t<T>;
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
