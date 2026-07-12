// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

struct zero_t;

struct nzero_t : broadcastable_base<nzero_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr nzero_t() noexcept = default;

    template <typename T>
    requires requires { -static_cast<T>(0); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this nzero_t) noexcept {
        return -static_cast<T>(0);
    }

    consteval zero_t operator-(this nzero_t) noexcept;
};

struct zero_t : broadcastable_base<zero_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr zero_t() noexcept = default;

    template <typename T>
    requires requires { static_cast<T>(0); }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this zero_t) noexcept {
        return static_cast<T>(0);
    }

    consteval nzero_t operator-(this zero_t) noexcept { return nzero_t{}; }
};

consteval zero_t nzero_t::operator-(this nzero_t) noexcept {
    return zero_t{};
}

inline constexpr zero_t zero{};

template <typename T>
requires explicitly_convertible_to<zero_t, T>
inline constexpr auto zero_v = static_cast<T>(zero);

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
