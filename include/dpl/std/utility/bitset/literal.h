// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#include "dpl/std/utility/apply.h"
#include "dpl/std/utility/bitset/integral_bitset.h"
#include "dpl/std/utility/bitset/large_bitset.h"

#if !DPL_MODULES
#  include "dpl/std/details/bitset.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

inline namespace bit_literals {

DPL_EXPORT template <details::utility::bitstr S>
DPL_NODISCARD consteval auto operator""_bits() noexcept {
    static_assert(S.data[0] == '0');
    static_assert(S.radix() == 2 || S.radix() == 16 || S.radix() == 8);
    static_assert([]() {
        constexpr auto radix = S.radix();
        if constexpr (radix == 2) {
            for (auto val : S) {
                if (val != '0' && val != '1') {
                    return false;
                }
            }
        } else if constexpr (radix == 16) {
            for (auto val : S) {
                if (!(val >= 'a' && val <= 'f') &&
                    !(val >= 'A' && val <= 'F') &&
                    !(val >= '0' && val <= '9')) {
                    return false;
                }
            }
        } else {
            for (auto val : S) {
                if (val < '0' || val > '7') {
                    return false;
                }
            }
        }
        return true;
    }());

    constexpr bool is_bin = S.radix() == 2;
    constexpr bool is_hex = S.radix() == 16;

    bitset<S.size()> output;
    if constexpr (is_bin) {
        for (auto idx = 0zu; auto const val : S) {
            output.set(idx++, val == '1');
        }
    } else if constexpr (is_hex) {
        for (auto idx = 0zu; auto const val : S) {
            auto const set = bitset<4>((val >= '0' && val <= '9') ? val - '0'
                    : (val >= 'a' && val <= 'f') ? 10 + (val - 'a')
                                                 : 10 + (val - 'A'));
            __DPL apply(
                [&](auto... bs) { (..., output.set(idx++, bs)); }, set);
        }
    } else {
        for (auto idx = 0zu; auto const val : S) {
            auto const set = bitset<3>(val - '0');
            __DPL apply(
                [&](auto... bs) { (..., output.set(idx++, bs)); }, set);
        }
    }

    return output;
}

} // namespace bit_literals

DPL_DEFAULT_NAMESPACE_END
