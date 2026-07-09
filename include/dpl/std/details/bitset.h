// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: always_keep

#if !DPL_MODULES
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/bit_width.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/has_single_bit.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
DPL_EXPORT template <size_t W>
class alignas(W / __DPL char_bit_v) bitset;

DPL_EXPORT namespace details::utility {
template <typename T>
inline constexpr bool is_bitset_v = false;
template <typename T>
inline constexpr bool is_bitset_v<T const> = is_bitset_v<T>;
template <typename T>
inline constexpr bool is_bitset_v<T volatile> = is_bitset_v<T>;
template <typename T>
inline constexpr bool is_bitset_v<T const volatile> = is_bitset_v<T>;
template <size_t W>
inline constexpr bool is_bitset_v<bitset<W>> = true;

template <typename T>
concept bitset_type = is_bitset_v<T>;

template <typename T>
concept bitset_constant_like = requires { T::value; } &&
    bitset_type<decltype(T::value)> && convertible_to<T, decltype(T::value)> &&
    equality_comparable_with<T, decltype(T::value)> && (T() == T::value) &&
    (static_cast<decltype(T::value)>(T()) == T::value);

consteval size_t ceil_pow2(size_t val) noexcept {
    return 1zu << (__DPL bit_width(val) - __DPL has_single_bit(val));
}

template <size_t W>
struct bitset_storage {
    using underlying_type = size_t[W / (sizeof(size_t) * __DPL char_bit_v) +
        (W % (sizeof(size_t) * __DPL char_bit_v) != 0)];
    underlying_type storage_;
};

template <size_t W>
requires requires { typename bit_type_t<utility::ceil_pow2(W)>; }
struct bitset_storage<W> {
    using underlying_type = bit_type_t<utility::ceil_pow2(W)>;
    underlying_type value_;
};

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

} // namespace details::utility

DPL_DEFAULT_NAMESPACE_END