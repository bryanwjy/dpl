// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#if !DPL_MODULES
#  include "dpl/core/concepts/common_class_with.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/simd_lane_type.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/type_traits/common_abi.h"
#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename T, typename L, typename R, typename A = common_abi_t<L, R>>
concept canonical_bitwise_result =
    simd_vector<T> && same_as<typename L::value_type, typename T::value_type> &&
    same_as<typename R::value_type, typename T::value_type> &&
    common_abi_with<A, typename T::abi_type> &&
    same_as<A, typename T::abi_type>;

template <typename T, typename L, typename R, typename A = common_abi_t<L, R>>
concept canonical_bitwise_mask = simd_mask<T> &&
    common_size_with<simd_lane_type_t<L>, simd_lane_type_t<T>> &&
    common_size_with<simd_lane_type_t<R>, simd_lane_type_t<T>> &&
    common_abi_with<A, typename T::abi_type> &&
    same_as<A, typename T::abi_type>;

template <typename T, typename In, typename A = typename In::abi_type>
concept canonical_bitshift_result = common_class_with<T, In> &&
    same_as<simd_lane_type_t<T>, simd_lane_type_t<In>> &&
    common_abi_with<A, typename T::abi_type> &&
    same_as<A, typename T::abi_type>;

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
