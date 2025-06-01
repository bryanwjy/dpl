// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/bitset.h"
#include "dpl/concepts.h"
#include "dpl/native.h"
#include "dpl/numeric.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::mask {

template <size_t W, typename T>
inline constexpr auto all_bits_v = numeric::all_bits_v<T>;

template <size_t W>
inline constexpr auto all_bits_v<W, unsigned char> =
    (numeric::all_bits_v<T> >> (CHAR_BIT - W));

template <abi_tag Abi, simd_element T>
inline constexpr auto all_bits_for_v =
    all_bits_v<element_count<Abi, T>, bit_type_t<element_count<Abi, T>>>;

} // namespace details::mask

DPL_EXPORT_BEGIN

template <size_t W, bit_type_t<W> V>
struct __DPL_PUBLIC_TEMPLATE basic_mask {
    using value_type = bit_type_t<W>;
    using type = basic_mask;
    static_assert(!(
        V & ~((static_cast<value_type>(1) << W) - static_cast<value_type>(1))));
    static constexpr size_t width = W;
    static constexpr value_type value = V;
    __DPL_HIDE_FROM_ABI constexpr operator value_type(
        this basic_mask) noexcept {
        return value;
    }
    __DPL_HIDE_FROM_ABI constexpr value_type operator+(
        this basic_mask) noexcept {
        return value;
    }
    __DPL_HIDE_FROM_ABI static constexpr value_type operator()() noexcept {
        return value;
    }

    __DPL_HIDE_FROM_ABI consteval basic_mask() noexcept = default;

    template <abi_tag Abi, simd_element T>
    requires (element_count<Abi, T> == width && value == 0)
    __DPL_HIDE_FROM_ABI constexpr basic_mask(Abi, zero_t<T>) noexcept {}

    template <abi_tag Abi, simd_element T>
    requires (element_count<Abi, T> == width &&
        value == details::mask::all_bits_v<value_type>)
    __DPL_HIDE_FROM_ABI constexpr basic_mask(Abi, all_t<T>) noexcept {}

    template <size_t W2>
    __DPL_HIDE_FROM_ABI constexpr basic_mask(
        basic_mask<W2, static_cast<bit_type_t<W2>>(value)>) noexcept {}

    template <simd_element T>
    __DPL_HIDE_FROM_ABI constexpr bool operator==(
        this basic_mask, all_t<T>) noexcept {
        return value == details::mask::all_bits_v<T>;
    }

    template <simd_element T>
    __DPL_HIDE_FROM_ABI constexpr bool operator!=(
        this basic_mask, all_t<T>) noexcept {
        return value != details::mask::all_bits_v<T>;
    }

    template <simd_element T>
    __DPL_HIDE_FROM_ABI constexpr bool operator==(
        this basic_mask, zero_t<T>) noexcept {
        return value == 0;
    }

    template <simd_element T>
    __DPL_HIDE_FROM_ABI constexpr bool operator!=(
        this basic_mask, zero_t<T>) noexcept {
        return value != 0;
    }

    template <size_t W2, bit_type_t<W2> V2>
    __DPL_HIDE_FROM_ABI constexpr bool operator==(
        this basic_mask, basic_mask<W2, V2>) noexcept {
        return V2 == value;
    }

    template <size_t W2, bit_type_t<W2> V2>
    __DPL_HIDE_FROM_ABI constexpr bool operator!=(
        this basic_mask, basic_mask<W2, V2>) noexcept {
        return V2 != value;
    }

    template <size_t W2, bit_type_t<W2> V2>
    __DPL_HIDE_FROM_ABI constexpr bool operator<(
        this basic_mask, basic_mask<W2, V2>) noexcept {
        return V2 < value;
    }

    template <size_t W2, bit_type_t<W2> V2>
    __DPL_HIDE_FROM_ABI constexpr bool operator<=(
        this basic_mask, basic_mask<W2, V2>) noexcept {
        return V2 <= value;
    }

    template <size_t W2, bit_type_t<W2> V2>
    __DPL_HIDE_FROM_ABI constexpr bool operator>(
        this basic_mask, basic_mask<W2, V2>) noexcept {
        return V2 > value;
    }

    template <size_t W2, bit_type_t<W2> V2>
    __DPL_HIDE_FROM_ABI constexpr bool operator>=(
        this basic_mask, basic_mask<W2, V2>) noexcept {
        return V2 >= value;
    }
};

template <abi_tag Abi, simd_element T>
explicit basic_mask(Abi, all_t<T>)
    -> basic_mask<element_count<Abi, T>, details::mask::all_bits_for_v<Abi, T>>;
template <abi_tag Abi, simd_element T>
explicit basic_mask(Abi, zero_t<T>) -> basic_mask<element_count<Abi, T>, 0>;

template <bit_type_t<1> V>
using mask1 = basic_mask<1, V>;
template <bit_type_t<2> V>
using mask2 = basic_mask<2, V>;
template <bit_type_t<4> V>
using mask4 = basic_mask<4, V>;
template <bit_type_t<8> V>
using mask8 = basic_mask<8, V>;
template <bit_type_t<16> V>
using mask16 = basic_mask<16, V>;

template <bit_type_t<1> V>
inline constexpr mask1 mask1_v{};
template <bit_type_t<2> V>
inline constexpr mask2 mask2_v{};
template <bit_type_t<4> V>
inline constexpr mask4 mask4_v{};
template <bit_type_t<8> V>
inline constexpr mask8 mask8_v{};
template <bit_type_t<16> V>
inline constexpr mask16 mask16_v{};

template <size_t W, size_t W2, bit_type_t<W2> V>
consteval auto truncate_to(basic_mask<W2, V>) noexcept {
    return basic_mask<W, V & ((1 << W) - 1)>{};
}

DPL_EXPORT_END

DPL_DEFAULT_NAMESPACE_END
