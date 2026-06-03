// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_class.h"
#include "dpl/core/concepts/simd_element_for.h"
#include "dpl/core/concepts/simd_lane_representation.h"
#include "dpl/core/concepts/simd_lane_type.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

namespace internal {
template <typename>
inline constexpr bool is_canonical_simd = false;
template <simd_abi A, simd_element_for<A> E>
inline constexpr bool is_canonical_simd<basic_vector<E, A>> = true;
template <typename>
inline constexpr bool is_canonical_mask = false;
template <simd_abi A, simd_element_for<A> E>
inline constexpr bool is_canonical_mask<basic_mask<E, A>> = true;
template <typename T>
concept canonical_simd_specialization = is_canonical_simd<T>;
template <typename T>
concept canonical_mask_specialization = is_canonical_mask<T>;

} // namespace internal

namespace atom {
template <typename T>
concept canonical_class = (internal::canonical_simd_specialization<T> ||
                              internal::canonical_mask_specialization<T>) &&
    requires { typename simd_lane_representation_t<T>; } &&
    same_as<simd_lane_representation_t<T>, simd_lane_type_t<T>>;
} // namespace atom

DPL_EXPORT template <typename T>
concept canonical_class = simd_class<T> && atom::canonical_class<T>;

DPL_EXPORT template <typename T>
concept canonical_vector = simd_vector<T> && canonical_class<T>;

DPL_EXPORT template <typename T>
concept canonical_mask = simd_mask<T> && canonical_class<T>;

DPL_EXPORT template <typename T>
concept extended_class = simd_class<T> && !canonical_class<T>;

DPL_EXPORT template <typename T>
concept extended_vector = simd_vector<T> && extended_class<T>;

DPL_EXPORT template <typename T>
concept extended_mask = simd_mask<T> && extended_class<T>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
