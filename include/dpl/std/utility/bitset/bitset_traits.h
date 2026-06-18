// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#include "dpl/std/utility/structured_bindings.h"

#if !DPL_MODULES
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/type_traits/is_scalar.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <size_t W>
class alignas(W / __DPL char_bit_v) bitset;

DPL_EXPORT template <size_t W>
struct tuple_size<bitset<W>> : size_constant<W> {};

DPL_EXPORT template <size_t I, size_t W>
struct tuple_element<I, bitset<W>> {
    using type DPL_NODEBUG = bool;
};

DPL_EXPORT template <size_t I, size_t W>
requires requires { typename bitset<W>; } &&
    is_integral_v<typename bitset<W>::underlying_type>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr auto get(bitset<W> value) noexcept {
    return value[I];
}

DPL_EXPORT template <size_t I, size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr auto get(
    bitset<W> const& value) noexcept {
    return value[I];
}

template <typename T>
inline constexpr bool is_bitset_v = false;
template <typename T>
inline constexpr bool is_bitset_v<T const> = is_bitset_v<T>;
template <typename T>
inline constexpr bool is_bitset_v<T volatile> = is_bitset_v<T>;
template <typename T>
inline constexpr bool is_bitset_v<T const volatile> = is_bitset_v<T>;
template <size_t W>
inline constexpr bool is_bitset_v<bitset<W>> = true;

DPL_DEFAULT_NAMESPACE_END
