// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_class.h"
#include "dpl/core/concepts/simd_element.h"

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
inline constexpr bool is_basic_simd = false;
template <simd_element E, simd_abi A>
inline constexpr bool is_basic_simd<basic_simd<E, A>> = true;
template <typename>
inline constexpr bool is_simd_mask = false;
template <simd_element E, simd_abi A>
inline constexpr bool is_simd_mask<basic_simd_mask<E, A>> = true;
template <typename T>
concept basic_simd_specialization = is_basic_simd<T>;
template <typename T>
concept simd_mask_specialization = is_simd_mask<T>;

} // namespace internal

DPL_EXPORT template <typename T>
struct basic_element {};

DPL_EXPORT template <typename T>
using basic_element_t = typename basic_element<T>::type;

DPL_EXPORT template <typename T>
concept basic_simd_element = simd_element<T> && requires {
    typename basic_element_t<T>;
    requires same_as<basic_element_t<T>, T>;
};

namespace atom {
template <typename T>
concept basic_simd_class = (internal::basic_simd_specialization<T> ||
                               internal::simd_mask_specialization<T>) &&
    basic_simd_element<simd_element_type_t<T>>;
}

DPL_EXPORT template <typename T>
concept basic_simd_class = simd_class<T> && atom::basic_simd_class<T>;

DPL_EXPORT template <typename T>
concept basic_simd_type = basic_simd_class<T> && atom::simd_type<T>;

DPL_EXPORT template <typename T>
concept basic_simd_mask_type = basic_simd_class<T> && atom::simd_mask_type<T>;

DPL_EXPORT template <enumeration T>
struct basic_element<T> {
    using type = T;
};

DPL_EXPORT template <floating_point T>
requires (sizeof(T) == sizeof(int16))
struct basic_element<T> {
#if DPL_SUPPORTS_FLOAT16
    using type = decltype(0.0f16);
#endif
};

DPL_EXPORT template <brain_float T>
requires (sizeof(T) == sizeof(int16))
struct basic_element<T> {
#if DPL_SUPPORTS_BFLOAT16
    using type = decltype(0.0bf16);
#endif
};

DPL_EXPORT template <floating_point T>
requires (sizeof(T) == sizeof(float))
struct basic_element<T> {
    using type DPL_NODEBUG = float;
};

DPL_EXPORT template <floating_point T>
requires (sizeof(T) == sizeof(double))
struct basic_element<T> {
    using type DPL_NODEBUG = double;
};

DPL_EXPORT template <signed_integral T>
requires (sizeof(T) <= sizeof(int64))
struct basic_element<T> {
    using type DPL_NODEBUG = make_signed_t<bit_type_t<sizeof(T) * char_bit_v>>;
};
DPL_EXPORT template <unsigned_integral T>
requires (sizeof(T) <= sizeof(int64))
struct basic_element<T> {
    using type DPL_NODEBUG = bit_type_t<sizeof(T) * char_bit_v>;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
