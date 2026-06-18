// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#include "dpl/std/utility/apply.h"
#include "dpl/std/utility/bitset/integral_bitset.h"
#include "dpl/std/utility/bitset/large_bitset.h"
#include "dpl/std/utility/structured_bindings.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace internal::bitset {
// Not exported
template <size_t W>
struct bitstr {
    char data[W + 1];

    static consteval size_t size() noexcept { return W; }
    consteval char const* begin() const noexcept { return data + W; }
    consteval char const* end() const noexcept { return data + 2; }
    consteval unsigned radix() const {
        char prefix = data[1];
        bool is_bin = (prefix == 'b' || prefix == 'B');
        bool is_hex = (prefix == 'x' || prefix == 'X');
        return is_bin ? 2 : is_hex ? 16 : 8;
    }
};

} // namespace internal::bitset

namespace bit_literals {

DPL_EXPORT template <internal::bitset::bitstr S>
DPL_NODISCARD consteval auto operator""_bits() noexcept {
    bitset<S.size()> output;
    static_assert(S.data[0] == '0');
    static_assert([]() {
        constexpr auto radix = S.radix();
        if constexpr (radix == 2) {
            for (auto val : S.data) {
                if (val != '0' && val != '1') {
                    return false;
                }
            }
        } else if constexpr (radix == 16) {
            for (auto val : S.data) {
                if (!(val >= 'a' && val <= 'f') &&
                    !(val >= 'A' && val <= 'F')) {
                    return false;
                }
            }
        } else {
            for (auto val : S.data) {
                if (val < '0' || val > '7') {
                    return false;
                }
            }
        }
        return true;
    }());

    constexpr bool is_bin = S.radix() == 2;
    constexpr bool is_hex = S.radix() == 16;
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
