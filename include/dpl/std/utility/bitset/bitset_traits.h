// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#if !DPL_MODULES
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/details/bitset.h"
#  include "dpl/std/type_traits/is_scalar.h"
#  include "dpl/std/type_traits/structured_bindings.h"
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
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr auto get(
    bitset<W> const& value) noexcept {
    return value[I];
}

DPL_DEFAULT_NAMESPACE_END
