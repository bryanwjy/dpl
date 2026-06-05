
// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/canonical.h"
#include "dpl/core/concepts/extended.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/simd_traits.h"
#  include "dpl/std/utility/to_underlying.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT enum class operation_category : unsigned {
    lane_agnostic = 1u << 0,
    lane_reduction = 1u << 1,
    lane_permutation = 1u << 2,
    lane_conversion = 1u << 3,
    structural_transformation = 1u << 4,
    all = ((1u << 5) - 1),
    none = 0u,
    default_category = lane_agnostic,
};

DPL_EXPORT consteval operation_category operator|(
    operation_category lhs, operation_category rhs) noexcept {
    return static_cast<operation_category>(
        __DPL to_underlying(lhs) | __DPL to_underlying(rhs));
}

DPL_EXPORT consteval operation_category operator&(
    operation_category lhs, operation_category rhs) noexcept {
    return static_cast<operation_category>(
        __DPL to_underlying(lhs) & __DPL to_underlying(rhs));
}

DPL_EXPORT consteval bool operator==(
    operation_category lhs, operation_category rhs) noexcept {
    return __DPL to_underlying(lhs) == __DPL to_underlying(rhs);
}

DPL_EXPORT consteval bool operator!=(
    operation_category lhs, operation_category rhs) noexcept {
    return !(lhs == rhs);
}

template <typename T, operation_category C>
concept decayable_simd_for = canonical_simd_type<T> ||
    (extended_simd_type<T> && ((simd_traits<T>::decay_policy & C) == C));

template <typename T, operation_category C>
concept decayable_vector_for = decayable_simd_for<T, C> && simd_vector<T>;

template <typename T, operation_category C>
concept decayable_mask_for = decayable_simd_for<T, C> && simd_mask<T>;

template <operation_category C, typename... T>
concept all_decayable = (... && decayable_simd_for<T, C>);

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
