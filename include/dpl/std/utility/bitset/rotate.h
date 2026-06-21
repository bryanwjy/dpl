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
    // TODO: optimize for large sets
    count %= W;
    if (count == 0)
        return val;

    if (count > 0) {
        return (val << count) | (val >> (W - count));
    }

    return (val >> -count) | (val << (W + count));
}

DPL_EXPORT template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bitset<W> rotr(bitset<W> const& val, int count) noexcept {
    // TODO: optimize for large sets
    count %= W;
    if (count == 0)
        return val;

    if (count > 0) {
        return (val >> count) | (val << (W - count));
    }

    return (val << -count) | (val >> (W + count));
}

DPL_DEFAULT_NAMESPACE_END
