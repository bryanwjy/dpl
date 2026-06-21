// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#include "dpl/std/utility/bitset/bitset_traits.h"
#include "dpl/std/utility/bitset/concepts.h"
#include "dpl/std/utility/bitset/storage.h"
#include "dpl/std/utility/bitset/to_underlying.h"
#include "dpl/std/utility/structured_bindings.h"

#if !DPL_MODULES
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/type_traits/is_base_of.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int popcount(bitset<W> const& val) noexcept {
    if constexpr (integral_bitset_type<bitset<W>>) {
        return __DPL popcount(__DPL to_underlying(val));
    } else {
        static_assert(is_base_of_v<details::bitset::storage<W>, bitset<W>>);
        auto const& base = (details::bitset::storage<W> const&)val;
        auto sum = 0zu;
        for (auto const val : base.storage_) {
            sum += __DPL popcount(val);
        }

        return sum;
    }
}

DPL_DEFAULT_NAMESPACE_END
