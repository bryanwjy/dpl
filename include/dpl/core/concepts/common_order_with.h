// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/arithmetic_type.h"
#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_float_with.h"
#include "dpl/core/concepts/common_integral_with.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/concepts/totally_ordered.h"
#  include "dpl/std/type_traits/underlying_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace internal {
/**
 * Enum is naturally ordered when it is totally ordered
 * and there are no comparison operations found via ADL
 */
template <typename T>
concept naturally_ordered_enum =
    enumeration<T> && totally_ordered<T> && requires(T val) {
        requires !requires { operator<(val, val); };
        requires !requires { operator>(val, val); };
        requires !requires { operator<=(val, val); };
        requires !requires { operator>=(val, val); };
        requires !requires { operator!=(val, val); };
        requires !requires { operator==(val, val); };
        requires !requires { operator<=>(val, val); };
    };

/**
 * Enum type, T, is comparable to an integral type, U, iff
 * U is a common integral with the underlying_type of T and
 * T is totally ordered with U and there are no comparison
 * operations found via ADL
 */
template <typename T, typename U>
concept enum_comparable = (same_as<T, U> && naturally_ordered_enum<T>) ||
    (enumeration<T> && common_integral_with<U, underlying_type_t<T>> &&
        totally_ordered_with<T, U> && requires(T lhs, U rhs) {
            requires !requires { operator<(lhs, rhs); };
            requires !requires { operator>(lhs, rhs); };
            requires !requires { operator<=(lhs, rhs); };
            requires !requires { operator>=(lhs, rhs); };
            requires !requires { operator!=(lhs, rhs); };
            requires !requires { operator==(lhs, rhs); };
            // Both sides need to be tested to ensure neither is synthesized
            // by the other
            requires !requires { operator<=>(lhs, rhs); };
            requires !requires { operator<=>(rhs, lhs); };
        });

template <typename T, typename U>
concept enum_common_order_with = (enumeration<T> && enum_comparable<T, U>) ||
    (enumeration<U> && enum_comparable<U, T>);

} // namespace internal

DPL_EXPORT template <typename A, typename B>
concept common_order_with = sizeof(A) == sizeof(B) &&
    ((arithmetic_type<A> && arithmetic_type<B> &&
         (same_as<A, B> || common_float_with<A, B> ||
             common_integral_with<A, B>)) ||
        internal::enum_common_order_with<A, B>);

DPL_EXPORT template <typename A, typename B>
concept common_order_simd_with = simd_common_abi_with<A, B> &&
    common_order_with<simd_element_type_t<A>, simd_element_type_t<B>>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
