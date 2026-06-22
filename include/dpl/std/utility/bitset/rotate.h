// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#include "dpl/std/utility/bitset/integral_bitset.h"
#include "dpl/std/utility/bitset/large_bitset.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bitset<W> rotl(bitset<W> const& val, int count) noexcept {
    count %= static_cast<int>(W);
    count += count < 0 ? static_cast<int>(W) : 0;
    if (count == 0)
        return val;

    return (val << count) | (val >> (W - count));
}

DPL_EXPORT template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bitset<W> rotr(bitset<W> const& val, int count) noexcept {
    count %= static_cast<int>(W);
    count += count < 0 ? static_cast<int>(W) : 0;
    if (count == 0)
        return val;

    return (val >> count) | (val << (W - count));
}

DPL_DEFAULT_NAMESPACE_END
