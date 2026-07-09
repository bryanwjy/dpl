// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#include "dpl/std/utility/bitset/bitset_traits.h"
#include "dpl/std/utility/bitset/concepts.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <size_t W>
requires integral_bitset_type<bitset<W>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) constexpr auto to_underlying(
    bitset<W> val) noexcept {
    return __DPL bit_cast<typename bitset<W>::underlying_type>(val);
}

DPL_DEFAULT_NAMESPACE_END
