// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_element_representation.h"
#include "dpl/core/concepts/simd_mask.h"
#include "dpl/core/concepts/simd_vector.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <typename T>
struct simd_lane_representation {};

DPL_EXPORT template <typename T>
using simd_lane_representation_t = typename simd_lane_representation<T>::type;

DPL_EXPORT template <simd_vector T>
struct simd_lane_representation<T> :
    simd_element_representation<typename T::abi_type, typename T::value_type> {
};

DPL_EXPORT template <simd_mask T>
struct simd_lane_representation<T> :
    simd_lane_representation<typename T::simd_vector> {};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
