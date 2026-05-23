// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#if !DPL_MODULES
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/simd_lane_type.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/type_traits/common_abi.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
template <typename T, typename U, typename A = typename U::abi_type>
concept canonical_vector_bit = simd_vector<T> &&
    common_size_with<simd_lane_type_t<U>, simd_lane_type_t<T>> &&
    unsigned_integral<simd_lane_type_t<T>> &&
    same_abi_as<A, typename T::abi_type>;

template <typename T, typename U, typename A = typename U::abi_type>
concept extended_vector_bit = simd_vector<T> &&
    common_size_with<simd_lane_type_t<U>, simd_lane_type_t<T>> &&
    unsigned_integral<simd_lane_type_t<T>> &&
    common_abi_with<A, typename T::abi_type>;

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
