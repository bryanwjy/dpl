// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#include "dpl/std/utility/bitset/bitset_traits.h"
#include "dpl/std/utility/bitset/concepts.h"
#include "dpl/std/utility/bitset/storage.h"
#include "dpl/std/utility/bitset/to_underlying.h"
#include "dpl/std/utility/structured_bindings.h"
#include "dpl/std/utility/to_unsigned.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/extent.h"
#  include "dpl/std/type_traits/is_base_of.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bitset<W> byteswap(bitset<W> const& val) noexcept {
    if constexpr (integral_bitset_type<bitset<W>>) {
        constexpr auto chunk = sizeof(bitset<W>) * __DPL char_bit_v;
        constexpr auto padding = chunk - W;
        if constexpr (padding > 0) {
            return bitset<W>( __DPL byteswap(
                __DPL to_unsigned(__DPL to_underlying(val) << padding)));
        } else {
            return bitset<W>(__DPL byteswap(__DPL to_underlying(val)));
        }
    } else {
        static_assert(is_base_of_v<details::bitset::storage<W>, bitset<W>>);
        using type DPL_NODEBUG = typename bitset<W>::underlying_type;
        constexpr auto chunk = sizeof(size_t) * __DPL char_bit_v;
        constexpr auto tail_size = W % chunk;
        constexpr auto padding = tail_size > 0 ? chunk - tail_size : 0zu;
        auto ret = __DPL bit_cast<bitset<W + padding>>(val);
        ret <<= padding;
        auto& base = (details::bitset::storage<W + padding>&)ret;
        for (auto left = base.storage_,
                  right = base.storage_ + extent_v<type> - 1;
            left < right;) {
            auto const tmp = *left;
            *left++ = __DPL byteswap(*right);
            *right-- = __DPL byteswap(tmp);
        }

        return __DPL bit_cast<bitset<W>>(ret);
    }
}

DPL_DEFAULT_NAMESPACE_END
