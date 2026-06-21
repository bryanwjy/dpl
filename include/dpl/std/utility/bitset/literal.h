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
class digit_ptr {
public:
    consteval digit_ptr() noexcept : ptr{} {}
    consteval digit_ptr(char const* ptr) noexcept : ptr{ptr} {}
    consteval bool operator==(digit_ptr const&) const noexcept = default;
    consteval bool operator!=(digit_ptr const&) const noexcept = default;
    consteval bool operator<(digit_ptr const& other) const noexcept {
        return ptr > other.ptr;
    }

    consteval digit_ptr& operator++() noexcept { return --ptr, *this; }

    consteval digit_ptr operator++(int) noexcept { return ptr--; }

    consteval char operator*() const noexcept { return *ptr; }
    consteval char const* operator->() const noexcept { return ptr; }

    consteval decltype(static_cast<char*>(0) - static_cast<char*>(0)) operator-(
        digit_ptr const& other) const noexcept {
        return other.ptr - ptr;
    }

private:
    char const* ptr;
};

template <size_t W>
struct bitstr {
    char data[W];

    consteval digit_ptr begin() const noexcept { return data + W - 2; }
    consteval digit_ptr end() const noexcept { return data + (radix() != 8); }
    consteval unsigned radix() const {
        char prefix = data[1];
        bool is_bin = (prefix == 'b' || prefix == 'B');
        bool is_hex = (prefix == 'x' || prefix == 'X');
        return is_bin ? 2 : is_hex ? 16 : 8;
    }

    consteval bitstr(char const (&str)[W]) noexcept {
        for (auto i = 0zu; i < W; ++i) {
            data[i] = str[i];
        }
    }

    consteval size_t size() const noexcept {
        return (end() - begin()) * __DPL countr_zero(radix());
    }
};

} // namespace internal::bitset

namespace bit_literals {

DPL_EXPORT template <internal::bitset::bitstr S>
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
