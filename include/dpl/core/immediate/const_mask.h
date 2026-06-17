// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/constants/all_bits.h"
#include "dpl/core/immediate/constants/zero.h"
#include "dpl/core/immediate/immediate.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/type_traits/canonical_type.h"
#  include "dpl/core/type_traits/declarg.h"
#  include "dpl/core/type_traits/enable_const_mask.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/countl.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/utility/bitset.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
namespace internal {
template <size_t W>
struct mask_value {};
template <size_t W>
using mask_value_t DPL_NODEBUG = typename mask_value<W>::type;

#if DPL_SUPPORTS_INT128
#  define __DPL_MAX_BITS 16
#else
#  define __DPL_MAX_BITS 8
#endif

template <size_t W>
requires (W <= __DPL_MAX_BITS && __DPL has_single_bit(W))
struct mask_value<W> : bit_type<W> {};

template <size_t W>
requires (W > __DPL_MAX_BITS && __DPL has_single_bit(W))
struct mask_value<W> {
    using type DPL_NODEBUG = bitset<W>;
};

#undef __DPL_MAX_BITS
} // namespace internal

DPL_EXPORT template <size_t W, internal::mask_value_t<W> V>
struct const_mask;

DPL_EXPORT template <size_t W, internal::mask_value_t<W> V>
struct const_mask : const_mask_base<const_mask<W, V>> {

private:
    template <size_t W2, internal::mask_value_t<W2> V2>
    static constexpr bool explicit_convertible_from_v = W2 > V2 &&
        (__DPL countl_zero(bitset<W2>(V2)) < __DPL countl_zero(bitset<W>(V)) ||
            (V2 != static_cast<decltype(V2)>(-1zu) &&
                V != static_cast<decltype(V)>(-1zu)));

public:
    using value_type = internal::mask_value_t<W>;
    using type = const_mask;
    static constexpr size_t width = W;
    static constexpr value_type value = []() {
        if constexpr (integral<value_type>) {
            return (V & static_cast<value_type>((1ll << W) - 1));
        } else {
            return V;
        }
    }();

    __DPL_HIDE_FROM_ABI constexpr operator immediate<value>(
        this const_mask) noexcept {
        return imm<value>;
    }
    __DPL_HIDE_FROM_ABI constexpr operator value_type(
        this const_mask) noexcept {
        return value;
    }

    __DPL_HIDE_FROM_ABI static constexpr value_type operator()() noexcept {
        return value;
    }

    __DPL_HIDE_FROM_ABI constexpr operator bitset<W>(this const_mask) noexcept
    requires integral<value_type>
    {
        return bitset<W>(value);
    }

    __DPL_HIDE_FROM_ABI consteval const_mask() noexcept = default;

    template <different_from<const_mask> T>
    requires integral_constant_like<T> && (T::value == V)
    __DPL_HIDE_FROM_ABI constexpr const_mask(T) noexcept {}

    template <different_from<const_mask> T>
    requires bitset_constant_like<T> && (T::value == bitset<T::value.size()>(V))
    __DPL_HIDE_FROM_ABI constexpr const_mask(T) noexcept {}

    __DPL_HIDE_FROM_ABI constexpr const_mask(zero_t) noexcept
    requires (V == 0)
    {}

    __DPL_HIDE_FROM_ABI constexpr const_mask(all_bits_t) noexcept
    requires (__DPL popcount(V) == W)
    {}

    template <size_t W2, internal::mask_value_t<W2> V2>
    __DPL_HIDE_FROM_ABI explicit(explicit_convertible_from_v<W2,
        V2>) constexpr const_mask(const_mask<W2, V2>) noexcept {}

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator==(this const_mask, all_bits_t) noexcept {
        return __DPL popcount(value) == static_cast<int>(W);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator!=(this const_mask, all_bits_t) noexcept {
        return __DPL popcount(value) == static_cast<int>(W);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator==(this const_mask, zero_t) noexcept {
        return (value & static_cast<value_type>(W - 1)) == 0;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator!=(this const_mask, zero_t) noexcept {
        return (value & static_cast<value_type>(W - 1)) != 0;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr auto operator~(this const_mask) noexcept {
        if constexpr (integral<value_type>) {
            return const_mask<W, static_cast<value_type>(~bitset<W>(value))>{};
        } else {
            return const_mask<W, ~value>{};
        }
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr bool operator==(all_bits_t, const_mask) noexcept {
        return __DPL popcount(value) == static_cast<int>(W);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr bool operator!=(all_bits_t, const_mask) noexcept {
        return __DPL popcount(value) == static_cast<int>(W);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr bool operator==(zero_t, const_mask) noexcept {
        return (value & static_cast<value_type>(W - 1)) == 0;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    friend constexpr bool operator!=(zero_t, const_mask) noexcept {
        return (value & static_cast<value_type>(W - 1)) != 0;
    }

    template <size_t W2, bit_type_t<W2> V2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator==(this const_mask, const_mask<W2, V2>) noexcept {
        return const_mask<W2, V2>::value == value;
    }

    template <size_t W2, bit_type_t<W2> V2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool operator!=(this const_mask, const_mask<W2, V2>) noexcept {
        return const_mask<W2, V2>::value != value;
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr bool operator[](size_t idx) noexcept {
        return (static_cast<value_type>(1zu << idx) & value) > 0;
    }

    friend consteval auto all_of(const_mask val) noexcept {
        return val == all_bits_t{};
    }

    friend consteval auto any_of(const_mask val) noexcept {
        return val != zero_t{};
    }

    friend consteval auto none_of(const_mask val) noexcept {
        return val == zero_t{};
    }

    friend consteval auto some_of(const_mask val) noexcept {
        return any_of(val) && !all_of(val);
    }

    friend consteval auto popcount(const_mask) noexcept {
        return __DPL popcount(value);
    }

    friend consteval auto rotl(const_mask) noexcept {
        if constexpr (integral<value_type>) {
            return __DPL rotl(bitset<W>(value));
        } else {
            return __DPL rotl(value);
        }
    }

    friend consteval auto rotr(const_mask) noexcept {
        if constexpr (integral<value_type>) {
            return __DPL rotr(bitset<W>(value));
        } else {
            return __DPL rotr(value);
        }
    }

    friend consteval auto countr_zero(const_mask) noexcept {
        if constexpr (integral<value_type>) {
            return __DPL countr_zero(bitset<W>(value));
        } else {
            return __DPL countr_zero(value);
        }
    }

    friend consteval auto countr_one(const_mask) noexcept {
        if constexpr (integral<value_type>) {
            return __DPL countr_one(bitset<W>(value));
        } else {
            return __DPL countr_one(value);
        }
    }

    friend consteval auto countl_one(const_mask) noexcept {
        if constexpr (integral<value_type>) {
            return __DPL countl_one(bitset<W>(value));
        } else {
            return __DPL countl_one(value);
        }
    }

    friend consteval auto countl_zero(const_mask) noexcept {
        if constexpr (integral<value_type>) {
            return __DPL countl_zero(bitset<W>(value));
        } else {
            return __DPL countl_zero(value);
        }
    }
};

template <typename C, auto>
struct make_const_mask {};

template <typename C, auto V>
using make_const_mask_t DPL_NODEBUG = typename make_const_mask<C, V>::type;

template <fixed_width_simd_type C, auto V>
requires explicitly_convertible_to<decltype(V),
    internal::mask_value_t<simd_abi_traits<C>::size>>
struct make_const_mask<C, V> {
private:
    static constexpr auto width = simd_abi_traits<C>::size();

public:
    using type DPL_NODEBUG =
        const_mask<width, static_cast<internal::mask_value_t<width>>(V)>;
};

DPL_EXPORT template <typename T>
concept mask_constant_like =
    (is_integral_v<decltype(T::value)> || bitset_type<decltype(T::value)>) &&
    !is_same_v<bool, remove_const_t<decltype(T::value)>> &&
    convertible_to<T, decltype(T::value)> &&
    equality_comparable_with<T, decltype(T::value)> &&
    bool_constant<T() == T::value>::value &&
    bool_constant<static_cast<decltype(T::value)>(T()) == T::value>::value;

template <typename M, typename T>
concept const_mask_from_constant = mask_constant_like<M> && requires(M mask) {
    typename make_const_mask_t<T, M::value>;
} && explicitly_convertible_to<M, make_const_mask_t<T, M::value>>;

DPL_EXPORT template <typename M, typename T>
concept const_mask_for = fixed_width_simd_type<T> &&
    (const_mask_from_constant<M, remove_cvref_t<T>> ||
        same_as<datapar::zero_t, M> || same_as<datapar::all_bits_t, M>);

DPL_EXPORT template <fixed_width_simd_type T, const_mask_for<T> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto to_compatible_const_mask(M mask) noexcept {
    return static_cast<make_const_mask_t<T, M::value>>(mask);
}

DPL_EXPORT template <fixed_width_simd_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto to_compatible_const_mask(datapar::zero_t) noexcept {
    return const_mask<simd_abi_traits<T>::size, 0>();
}

DPL_EXPORT template <fixed_width_simd_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto to_compatible_const_mask(datapar::all_bits_t) noexcept {
    constexpr auto V = static_cast<bit_type_t<simd_abi_traits<T>::size>>(-1);
    return const_mask<simd_abi_traits<T>::size, V>();
}

template <typename T>
concept const_mask_like = enable_const_mask<T> && integral_constant_like<T> &&
    unsigned_integral<typename T::value_type> &&
    requires { typename integral_constant<size_t, T::width>; } &&
    (T::width < sizeof(typename T::value_type) * char_bit_v);

template <typename T>
concept const_mask_convertible =
    integral_constant_like<T> || bitset_constant_like<T>;

template <size_t W, internal::mask_value_t<W> V>
consteval bitset<W> to_bitset(const_mask<W, V> self) noexcept {
    return static_cast<bitset<W>>(self);
}

namespace internal {
template <typename S, typename M>
struct launder_cmask {};
template <simd_type S, const_mask_for<canonical_type_t<S>> M>
struct launder_cmask<S, M> {
    using type DPL_NODEBUG =
        decltype(dx::to_compatible_const_mask<canonical_type_t<S>>(
            internal::declarg<M>()));
};
template <typename S, typename M>
using launder_cmask_t DPL_NODEBUG = typename launder_cmask<S, M>::type;
} // namespace internal
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
