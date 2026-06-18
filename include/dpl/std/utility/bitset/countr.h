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
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/type_traits/is_base_of.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countr_one(bitset<W> const& val) noexcept {
    if constexpr (integral_bitset_type<bitset<W>>) {
        return __DPL countr_one(__DPL to_underlying(val));
    } else {
        static_assert(is_base_of_v<details::bitset::storage<W>, bitset<W>>);
        using type DPL_NODEBUG = typename bitset<W>::underlying_type;
        constexpr auto chunk = sizeof(size_t) * __DPL char_bit_v;
        constexpr auto tail_size = W % chunk;
        constexpr auto padding = tail_size > 0 ? chunk - tail_size : 0zu;
        auto const& base = (details::bitset::storage<W> const&)val;
        for (auto result = 0zu; auto const val : base.storage_) {
            if (val != -1zu) {
                return result * chunk + __DPL countr_one(val);
            }

            ++result;
        }

        return W;
    }
}

template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countr_zero(bitset<W> const& val) noexcept {
    if constexpr (integral_bitset_type<bitset<W>>) {
        return __DPL countr_one(__DPL to_underlying(val));
    } else {
        static_assert(is_base_of_v<details::bitset::storage<W>, bitset<W>>);
        using type DPL_NODEBUG = typename bitset<W>::underlying_type;
        constexpr auto chunk = sizeof(size_t) * __DPL char_bit_v;
        constexpr auto tail_size = W % chunk;
        constexpr auto padding = tail_size > 0 ? chunk - tail_size : 0zu;
        auto const& base = (details::bitset::storage<W> const&)val;
        for (auto result = 0zu; auto const val : base.storage_) {
            if (val != 0zu) {
                return result * chunk + __DPL countr_zero(val);
            }

            ++result;
        }

        return W;
    }
}

DPL_DEFAULT_NAMESPACE_END
