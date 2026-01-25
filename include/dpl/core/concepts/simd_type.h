// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_element.h"
#include "dpl/core/concepts/simd_type.h"

#if !DPL_MODULE
#  include "dpl/core/fwd/basic.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/type_traits/is_constructible.h"
#  include "dpl/std/type_traits/is_convertible.h"
#  include "dpl/std/type_traits/is_enum.h"
#  include "dpl/std/type_traits/is_object.h"
#  include "dpl/std/type_traits/is_trivially_copyable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
inline constexpr bool enable_simd_type = false;
DPL_EXPORT template <simd_element T, simd_abi Abi>
inline constexpr bool enable_simd_type<basic_simd<T, Abi>> = true;

DPL_EXPORT template <typename T>
concept simd_type = enable_simd_type<T> && is_object_v<T> && semiregular<T> &&
    is_trivially_copyable_v<T> && requires {
        typename T::value_type;
        typename T::abi_type;
        requires simd_element<typename T::value_type>;
        requires simd_abi<typename T::abi_type>;
        requires sizeof(typename T::value_type) <= T::abi_type::size &&
                alignof(typename T::value_type) <= T::abi_type::alignment;
        requires alignof(T) >= T::abi_type::alignment;
    } && requires(T const val) {
        {
            +val
        } -> same_as<
            typename T::abi_type::template native_type<typename T::value_type>>;
    };

DPL_EXPORT template <simd_type T>
struct simd_element_type<T> {
    using type DPL_NODEBUG = typename T::value_type;
};

DPL_EXPORT template <typename T>
concept integral_simd = simd_type<T> && integral<simd_element_type_t<T>>;
DPL_EXPORT template <typename T>
concept signed_integral_simd =
    simd_type<T> && signed_integral<simd_element_type_t<T>>;
DPL_EXPORT template <typename T>
concept unsigned_integral_simd =
    simd_type<T> && unsigned_integral<simd_element_type_t<T>>;
DPL_EXPORT template <typename T>
concept floating_point_simd =
    simd_type<T> && floating_point<simd_element_type_t<T>>;
DPL_EXPORT template <typename T>
concept enum_simd = simd_type<T> && is_enum_v<simd_element_type_t<T>>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
