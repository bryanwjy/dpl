// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_class.h"
#include "dpl/core/concepts/simd_element.h"
#include "dpl/core/concepts/simd_lane_representation.h"
#include "dpl/core/concepts/simd_lane_type.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/type_traits/make_signed.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

namespace internal {
template <typename>
inline constexpr bool is_canonical_simd = false;
template <simd_element E, simd_abi A>
inline constexpr bool is_canonical_simd<basic_simd<E, A>> = true;
template <typename>
inline constexpr bool is_canonical_mask = false;
template <simd_element E, simd_abi A>
inline constexpr bool is_canonical_mask<basic_simd_mask<E, A>> = true;
template <typename T>
concept canonical_simd_specialization = is_canonical_simd<T>;
template <typename T>
concept canonical_mask_specialization = is_canonical_mask<T>;

} // namespace internal

namespace atom {
template <typename T>
concept basic_simd_class = (internal::canonical_simd_specialization<T> ||
                               internal::canonical_mask_specialization<T>) &&
    requires { typename simd_lane_representation_t<T>; } &&
    same_as<simd_lane_representation_t<T>, simd_lane_type_t<T>>;
} // namespace atom

DPL_EXPORT template <typename T>
concept basic_simd_class = simd_class<T> && atom::basic_simd_class<T>;

DPL_EXPORT template <typename T>
concept basic_simd_type = basic_simd_class<T> && atom::simd_type<T>;

DPL_EXPORT template <typename T>
concept basic_simd_mask_type = basic_simd_class<T> && atom::simd_mask_type<T>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
