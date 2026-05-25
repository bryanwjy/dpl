// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#if !DPL_MODULES
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/type_traits/common_abi.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
template <typename T, typename L, typename R, typename A = common_abi_t<L, R>>
concept canonical_logical_result = simd_mask<T> &&
    common_size_with<simd_lane_type_t<L>, simd_lane_type_t<T>> &&
    common_size_with<simd_lane_type_t<R>, simd_lane_type_t<T>> &&
    same_abi_as<A, typename T::abi_type>;

template <typename T, typename L, typename R, typename A = common_abi_t<L, R>>
concept extended_logical_result = simd_mask<T> &&
    common_size_with<simd_lane_type_t<L>, simd_lane_type_t<T>> &&
    common_size_with<simd_lane_type_t<R>, simd_lane_type_t<T>> &&
    common_abi_with<typename T::abi_type, A>;
} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
