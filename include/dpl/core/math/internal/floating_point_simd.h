// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_equivalence.h"
#  include "dpl/core/concepts/simd_traits.h"
#  include "dpl/core/concepts/simd_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
template <typename T>
concept floating_point_simd =
    simd_type<T> && floating_point<simd_lane_type_t<T>>;
template <typename T, typename A>
concept floating_point_simd_with_abi =
    simd_with_abi<T, A> && floating_point_simd<T>;
} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
