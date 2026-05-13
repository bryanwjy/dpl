// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_type.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
concept integral_simd = simd_type<T> && integral<simd_lane_type_t<T>>;
DPL_EXPORT template <typename T>
concept signed_integral_simd =
    simd_type<T> && signed_integral<simd_lane_type_t<T>>;
DPL_EXPORT template <typename T>
concept unsigned_integral_simd =
    simd_type<T> && unsigned_integral<simd_lane_type_t<T>>;
DPL_EXPORT template <typename T>
concept floating_point_simd =
    simd_type<T> && floating_point<simd_lane_type_t<T>>;
DPL_EXPORT template <typename T>
concept enumeration_simd = simd_type<T> && enumeration<simd_lane_type_t<T>>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
