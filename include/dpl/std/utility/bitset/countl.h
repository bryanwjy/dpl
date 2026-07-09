// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#include "dpl/std/utility/bitset/bitset_traits.h"
#include "dpl/std/utility/bitset/concepts.h"
#include "dpl/std/utility/bitset/to_underlying.h"
#include "dpl/std/utility/to_unsigned.h"

#if !DPL_MODULES
#  include "dpl/std/bit/countl.h"
#  include "dpl/std/type_traits/extent.h"
#  include "dpl/std/type_traits/is_base_of.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countl_one(bitset<W> const& val) noexcept {
    if constexpr (integral_bitset_type<bitset<W>>) {
        constexpr auto chunk = sizeof(bitset<W>) * __DPL char_bit_v;
        constexpr auto padding = chunk - W;
        if constexpr (padding > 0) {
            return __DPL countl_zero(__DPL to_underlying(~val)) - padding;
        } else {
            return __DPL countl_one(__DPL to_underlying(val));
        }
    } else {
        using type DPL_NODEBUG = typename bitset<W>::underlying_type;
        constexpr auto chunk = sizeof(size_t) * __DPL char_bit_v;
        constexpr auto tail_size = W % chunk;
        constexpr auto padding = tail_size > 0 ? chunk - tail_size : 0zu;

        details::utility::bitset_storage<W> const& base = val;
        auto const* ptr = base.storage_ + extent_v<type>;

        auto result = 0zu;
        if constexpr (tail_size > 0) {
            auto const count = __DPL countl_one(*--ptr << padding);
            if (count < tail_size) {
                return count;
            }
        }

        for (; ptr > base.storage_; ++result) {
            if (auto const val = *--ptr; val != -1zu) {
                return result * chunk + tail_size + __DPL countl_one(val);
            }
        }

        return W;
    }
}

DPL_EXPORT template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int countl_zero(bitset<W> const& val) noexcept {
    if constexpr (integral_bitset_type<bitset<W>>) {
        constexpr auto chunk =
            sizeof(typename bitset<W>::underlying_type) * __DPL char_bit_v;
        constexpr auto padding = chunk - W;
        if constexpr (padding > 0) {
            return __DPL countl_zero(__DPL to_underlying(val)) - padding;
        } else {
            return __DPL countl_zero(__DPL to_underlying(val));
        }
    } else {
        using type DPL_NODEBUG = typename bitset<W>::underlying_type;
        constexpr auto chunk = sizeof(size_t) * __DPL char_bit_v;
        constexpr auto tail_size = W % chunk;
        constexpr auto padding = tail_size > 0 ? chunk - tail_size : 0zu;

        details::utility::bitset_storage<W> const& base = val;
        auto const* ptr = base.storage_ + extent_v<type>;

        auto result = 0zu;
        if constexpr (tail_size > 0) {
            auto const count = __DPL countl_zero(*--ptr << padding);
            if (count < tail_size) {
                return count;
            }
        }

        for (; ptr > base.storage_; ++result) {
            if (auto const val = *--ptr; val != 0zu) {
                return result * chunk + tail_size + __DPL countl_zero(val);
            }
        }

        return W;
    }
}

DPL_DEFAULT_NAMESPACE_END
