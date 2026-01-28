// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/arithmetic_type.h"
#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_float_with.h"
#include "dpl/core/concepts/common_size_with.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace internal {
template <typename T, typename U>
concept enum_arithemtic_operable =
    (naturally_arithmetic_enum<T> && same_as<T, U>) ||
    (unscoped_enumeration<T> && integral<U> && common_size_with<T, U> &&
        requires(T lhs, U rhs) {
            requires !requires { operator+(lhs, rhs); } && requires {
                { lhs + rhs } -> underlying_type_of<T>;
            };
            requires !requires { operator-(lhs, rhs); } && requires {
                { lhs - rhs } -> underlying_type_of<T>;
            };
            requires !requires { operator*(lhs, rhs); } && requires {
                { lhs * rhs } -> underlying_type_of<T>;
            };
            requires !requires { operator-(lhs); } && requires {
                { -rhs } -> underlying_type_of<T>;
            };
        });

/**
 * An enum, T, is common arithmetic with itself iff T statisfies
 * naturally_arithmetic_enum.
 *
 * An unscoped enum, T, is common arithmetic with an integral U, if they
 * have the same size and no addition,subtraction,multiplication
 * operations between T and U and negation operation of T are found via ADL
 */
template <typename T, typename U>
concept enum_common_arithmetic_with =
    (unscoped_enumeration<T> && enum_arithemtic_operable<T, U>) ||
    (unscoped_enumeration<U> && enum_arithemtic_operable<U, T>);
} // namespace internal

/**
 * Arithmetic operations (excluding division for integers) may use the same
 * instruction, signed-ness do not matter. Enums are only allowed if arithmetic
 * enabled.
 */
DPL_EXPORT template <typename A, typename B>
concept common_arithmetic_with =
    common_size_with<A, B> && arithmetic_type<A> && arithmetic_type<B> &&
    (common_float_with<A, B> || (integral<A> && integral<B>) ||
        internal::enum_common_arithmetic_with<A, B>);

DPL_EXPORT template <typename A, typename B>
concept common_arithmetic_simd_with = simd_common_abi_with<A, B> &&
    common_arithmetic_with<simd_element_type_t<A>, simd_element_type_t<B>>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
