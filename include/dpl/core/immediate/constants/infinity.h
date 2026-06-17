// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"
#include "dpl/core/immediate/constants/exponent_bits.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT struct ninfinity_t;
DPL_EXPORT struct infinity_t : broadcastable_base<infinity_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr infinity_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this infinity_t) noexcept {
        return static_cast<T>(exponent_bits);
    }

    consteval auto operator-(this infinity_t) noexcept;
};
DPL_EXPORT using pinfinity_t = infinity_t;
DPL_EXPORT inline constexpr pinfinity_t pinfinity{};
DPL_EXPORT inline constexpr infinity_t infinity{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<infinity_t, T>
inline constexpr auto infinity_v = static_cast<T>(infinity);

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<pinfinity_t, T>
inline constexpr auto pinfinity_v = static_cast<T>(pinfinity);

DPL_EXPORT struct ninfinity_t : broadcastable_base<ninfinity_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr ninfinity_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this ninfinity_t) noexcept {
        return -static_cast<T>(exponent_bits);
    }

    consteval auto operator-(this ninfinity_t) noexcept;
};

DPL_EXPORT inline constexpr ninfinity_t ninfinity{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<ninfinity_t, T>
inline constexpr auto ninfinity_v = static_cast<T>(ninfinity);

consteval auto infinity_t::operator-(this infinity_t) noexcept {
    return ninfinity;
}

consteval auto ninfinity_t::operator-(this ninfinity_t) noexcept {
    return infinity;
}

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
