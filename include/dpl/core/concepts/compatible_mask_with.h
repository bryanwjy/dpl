// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_size_with.h"
#include "dpl/core/concepts/simd_mask.h"
#include "dpl/core/concepts/simd_vector.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename M, typename T>
concept compatible_mask_with = simd_mask<M> && simd_vector<T> &&
    common_abi_with<typename T::abi_type, typename M::abi_type> &&
    common_size_with<simd_lane_type_t<M>, simd_lane_type_t<T>>;
DPL_EXPORT template <typename M, typename T>
concept exact_mask_for = simd_mask<M> && simd_vector<T> &&
    same_abi_as<typename T::abi_type, typename M::abi_type> &&
    common_size_with<simd_lane_type_t<M>, simd_lane_type_t<T>>;
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
