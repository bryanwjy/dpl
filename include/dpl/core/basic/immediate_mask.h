// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/all_bits.h"
#include "dpl/core/basic/immediate.h"
#include "dpl/core/basic/zero.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/immediate_mask_like.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <size_t W, bit_type_t<W> V>
struct basic_immediate_mask;

DPL_EXPORT template <size_t W, bit_type_t<W> V>
inline constexpr bool enable_immediate_mask<basic_immediate_mask<W, V>> = true;

DPL_EXPORT template <size_t W, bit_type_t<W> V>
struct basic_immediate_mask {
    using value_type = bit_type_t<W>;
    using type = basic_immediate_mask;
    static constexpr size_t width = W;
    static constexpr value_type value =
        (V & static_cast<value_type>((1ll << W) - 1));
    __DPL_HIDE_FROM_ABI constexpr operator immediate<value>(
        this basic_immediate_mask) noexcept {
        return imm<value>;
    }
    __DPL_HIDE_FROM_ABI constexpr operator value_type(
        this basic_immediate_mask) noexcept {
        return value;
    }
    __DPL_HIDE_FROM_ABI constexpr value_type operator+(
        this basic_immediate_mask) noexcept {
        return value;
    }
    __DPL_HIDE_FROM_ABI static constexpr value_type operator()() noexcept {
        return value;
    }

    __DPL_HIDE_FROM_ABI consteval basic_immediate_mask() noexcept = default;

    template <integral auto Vin>
    requires (Vin == V)
    __DPL_HIDE_FROM_ABI constexpr basic_immediate_mask(
        immediate<Vin>) noexcept {}

    __DPL_HIDE_FROM_ABI constexpr basic_immediate_mask(zero_t) noexcept
    requires (V == 0)
    {}

    __DPL_HIDE_FROM_ABI constexpr basic_immediate_mask(all_bits_t) noexcept
    requires (__DPL countr_one(V) == __DPL char_bit_v * sizeof(V))
    {}

    template <size_t W2>
    __DPL_HIDE_FROM_ABI constexpr basic_immediate_mask(
        basic_immediate_mask<W2, static_cast<bit_type_t<W2>>(value)>) noexcept {
    }

    template <simd_abi A, simd_element_for<A> E>
    requires (simd_abi_traits<E, A>::size == W)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr operator basic_simd_mask<E, A>() const noexcept {
        return []<size_t... Is>(index_sequence<Is...>) {
            return basic_simd<E, A>(operator[](Is)...);
        }(iota_sequence<E, A>);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator==(this basic_immediate_mask, all_bits_t) noexcept {
        return __DPL popcount(value) == static_cast<int>(W);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator!=(this basic_immediate_mask, all_bits_t) noexcept {
        return __DPL popcount(value) == static_cast<int>(W);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator==(this basic_immediate_mask, zero_t) noexcept {
        return (value & static_cast<value_type>(W - 1)) == 0;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator!=(this basic_immediate_mask, zero_t) noexcept {
        return (value & static_cast<value_type>(W - 1)) != 0;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr auto operator~(this basic_immediate_mask) noexcept {
        constexpr auto mask = static_cast<value_type>((1 << W) - 1);
        constexpr auto nvalue = ~value;
        return basic_immediate_mask<W, nvalue & mask>{};
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr bool operator==(
        all_bits_t, basic_immediate_mask) noexcept {
        return __DPL popcount(value) == static_cast<int>(W);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr bool operator!=(
        all_bits_t, basic_immediate_mask) noexcept {
        return __DPL popcount(value) == static_cast<int>(W);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr bool operator==(zero_t, basic_immediate_mask) noexcept {
        return (value & static_cast<value_type>(W - 1)) == 0;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr bool operator!=(zero_t, basic_immediate_mask) noexcept {
        return (value & static_cast<value_type>(W - 1)) != 0;
    }

    template <size_t W2, bit_type_t<W2> V2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator==(
        this basic_immediate_mask, basic_immediate_mask<W2, V2>) noexcept {
        return basic_immediate_mask<W2, V2>::value == value;
    }

    template <size_t W2, bit_type_t<W2> V2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator!=(
        this basic_immediate_mask, basic_immediate_mask<W2, V2>) noexcept {
        return basic_immediate_mask<W2, V2>::value != value;
    }

    template <size_t W2, bit_type_t<W2> V2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator<(
        this basic_immediate_mask, basic_immediate_mask<W2, V2>) noexcept {
        return basic_immediate_mask<W2, V2>::value < value;
    }

    template <size_t W2, bit_type_t<W2> V2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator<=(
        this basic_immediate_mask, basic_immediate_mask<W2, V2>) noexcept {
        return basic_immediate_mask<W2, V2>::value <= value;
    }

    template <size_t W2, bit_type_t<W2> V2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator>(
        this basic_immediate_mask, basic_immediate_mask<W2, V2>) noexcept {
        return basic_immediate_mask<W2, V2>::value > value;
    }

    template <size_t W2, bit_type_t<W2> V2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator>=(
        this basic_immediate_mask, basic_immediate_mask<W2, V2>) noexcept {
        return basic_immediate_mask<W2, V2>::value >= value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr bool operator[](size_t idx) noexcept {
        return (static_cast<value_type>(1zu << idx) & value) > 0;
    }

    friend consteval auto all_of(basic_immediate_mask val) noexcept {
        return val == all_bits_t{};
    }

    friend consteval auto any_of(basic_immediate_mask val) noexcept {
        return val != zero_t{};
    }

    friend consteval auto none_of(basic_immediate_mask val) noexcept {
        return val == zero_t{};
    }

    friend consteval auto some_of(basic_immediate_mask val) noexcept {
        return any_of(val) && !all_of(val);
    }

    friend consteval auto popcount(basic_immediate_mask) noexcept {
        return __DPL popcount(value);
    }

    friend consteval auto countr_zero(basic_immediate_mask) noexcept {
        auto const count = __DPL countr_zero(value);
        return count < W ? count : W;
    }

    friend consteval auto countr_one(basic_immediate_mask) noexcept {
        auto const count = __DPL countr_one(value);
        return count < W ? count : W;
    }

    friend consteval auto countl_one(basic_immediate_mask) noexcept {
        constexpr auto mask = static_cast<value_type>(-1ll << width);
        auto const remainder = sizeof(value_type) * char_bit_v - width;
        return __DPL countl_one(value | mask) - remainder;
    }

    friend consteval auto countl_zero(basic_immediate_mask) noexcept {
        auto const offset = sizeof(value_type) * char_bit_v - W;
        return __DPL countl_zero(value) - offset;
    }
};

DPL_EXPORT template <size_t W, convertible_to<bit_type_t<W>> auto V>
using immediate_mask DPL_NODEBUG =
    basic_immediate_mask<W, static_cast<bit_type_t<W>>(V)>;

DPL_EXPORT template <typename C, auto>
struct make_immediate_mask {};

DPL_EXPORT template <typename C, auto V>
using make_immediate_mask_t DPL_NODEBUG =
    typename make_immediate_mask<C, V>::type;

DPL_EXPORT template <fixed_width_class C, auto V>
struct make_immediate_mask<C, V> {
    using type DPL_NODEBUG = immediate_mask<simd_abi_traits<C>::size, V>;
};

DPL_EXPORT template <typename M, typename T>
concept immediate_mask_for =
    fixed_width_class<T> && integral_constant_like<M> && requires(M mask) {
        typename basic_immediate_mask<simd_abi_traits<T>::size, M::value>;
        requires convertible_to<M,
            basic_immediate_mask<simd_abi_traits<T>::size, M::value>>;
    };

DPL_EXPORT template <fixed_width_class T, immediate_mask_for<T> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto to_immediate_mask(M mask) noexcept {
    return static_cast<
        basic_immediate_mask<simd_abi_traits<T>::size, M::value>>(mask);
}

template <fixed_width_class T, immediate_mask_for<T> M>
inline constexpr auto immediate_mask_v =
    decltype(datapar::to_immediate_mask<T>(M{}))::value;

DPL_EXPORT template <convertible_to<bit_type_t<1>> auto V>
using imm_mask1_t DPL_NODEBUG =
    basic_immediate_mask<1, static_cast<bit_type_t<1>>(V)>;
DPL_EXPORT template <convertible_to<bit_type_t<2>> auto V>
using imm_mask2_t DPL_NODEBUG =
    basic_immediate_mask<2, static_cast<bit_type_t<2>>(V)>;
DPL_EXPORT template <convertible_to<bit_type_t<4>> auto V>
using imm_mask4_t DPL_NODEBUG =
    basic_immediate_mask<4, static_cast<bit_type_t<4>>(V)>;
DPL_EXPORT template <convertible_to<bit_type_t<8>> auto V>
using imm_mask8_t DPL_NODEBUG =
    basic_immediate_mask<8, static_cast<bit_type_t<8>>(V)>;
DPL_EXPORT template <convertible_to<bit_type_t<16>> auto V>
using imm_mask16_t DPL_NODEBUG =
    basic_immediate_mask<16, static_cast<bit_type_t<16>>(V)>;

DPL_EXPORT template <convertible_to<bit_type_t<1>> auto V>
inline constexpr imm_mask1_t<V> imm_mask1{};
DPL_EXPORT template <convertible_to<bit_type_t<2>> auto V>
inline constexpr imm_mask2_t<V> imm_mask2{};
DPL_EXPORT template <convertible_to<bit_type_t<4>> auto V>
inline constexpr imm_mask4_t<V> imm_mask4{};
DPL_EXPORT template <convertible_to<bit_type_t<8>> auto V>
inline constexpr imm_mask8_t<V> imm_mask8{};
DPL_EXPORT template <convertible_to<bit_type_t<16>> auto V>
inline constexpr imm_mask16_t<V> imm_mask16{};

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
