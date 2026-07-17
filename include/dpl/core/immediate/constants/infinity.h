// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"
#include "dpl/core/immediate/constants/exponent_bits.h"

#if !DPL_MODULES
#  include "dpl/core/numbers/floating_point_traits.h"
#  include "dpl/std/concepts/convertible_to.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

struct ninfinity_t;

struct infinity_t : broadcastable_base<infinity_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr infinity_t() noexcept = default;

    template <floating_point_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this infinity_t) noexcept {
        if constexpr (floating_point_traits<T>::has_hidden_bit) {
            return __DPL bit_cast<T>(floating_point_traits<T>::exponent_mask);
        } else {
            return __DPL bit_cast<T>(
                (floating_point_traits<T>::exponent_mask >> 1) |
                floating_point_traits<T>::exponent_mask);
        }
    }

    consteval auto operator-(this infinity_t) noexcept;
};
using pinfinity_t = infinity_t;
inline constexpr pinfinity_t pinfinity{};
inline constexpr infinity_t infinity{};

template <typename T>
requires explicitly_convertible_to<infinity_t, T>
inline constexpr auto infinity_v = static_cast<T>(infinity);

template <typename T>
requires explicitly_convertible_to<pinfinity_t, T>
inline constexpr auto pinfinity_v = static_cast<T>(pinfinity);

struct ninfinity_t : broadcastable_base<ninfinity_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr ninfinity_t() noexcept = default;

    template <floating_point_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this ninfinity_t) noexcept {
        return -static_cast<T>(exponent_bits);
    }

    consteval auto operator-(this ninfinity_t) noexcept;
};

inline constexpr ninfinity_t ninfinity{};

template <typename T>
requires explicitly_convertible_to<ninfinity_t, T>
inline constexpr auto ninfinity_v = static_cast<T>(ninfinity);

consteval auto infinity_t::operator-(this infinity_t) noexcept {
    return ninfinity;
}

consteval auto ninfinity_t::operator-(this ninfinity_t) noexcept {
    return infinity;
}

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
