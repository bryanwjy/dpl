// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/bitset/bitset_traits.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
template <size_t W, size_t W2>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bitset<W> truncate(bitset<W2> const& val) noexcept {
    static_assert(W <= W2);
    return static_cast<bitset<W>>(val);
}
__DPL_DEFAULT_NAMESPACE_END
