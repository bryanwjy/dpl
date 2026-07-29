// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/bitset/bitset_traits.h"
#include "dpl/std/utility/bitset/concepts.h"
#include "dpl/std/utility/bitset/to_underlying.h"

#if !DPL_MODULES
#  include "dpl/std/bit/popcount.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int popcount(bitset<W> const& val) noexcept {
    if constexpr (integral_bitset_type<bitset<W>>) {
        return __DPL popcount(__DPL to_underlying(val));
    } else {
        details::utility::bitset_storage<W> const& base = val;
        auto sum = 0zu;
        for (auto const val : base.storage_) {
            sum += __DPL popcount(val);
        }

        return sum;
    }
}

__DPL_DEFAULT_NAMESPACE_END
