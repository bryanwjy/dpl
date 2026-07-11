// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/details/fwd.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/enable_simd_mask.h"
#  include "dpl/core/type_traits/enable_simd_vector.h"
#  include "dpl/std/concepts/different_from.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename T>
concept has_expression_result =
    (enable_simd_mask<T> || enable_simd_vector<T>) &&
    different_from<typename canonical_type<T>::type, T> &&
    requires { typename T::result_type; } &&
    different_from<typename T::result_type, T> &&
    (enable_simd_mask<typename T::result_type> ||
        enable_simd_vector<typename T::result_type>);
} // namespace datapar::internal
__DPL_DEFAULT_NAMESPACE_END
