// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/utility/to_underlying.h"
#  include "dpl/std/utility/to_unsigned.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::fmath {

enum class rounding_flags {
    to_nearest_int = 1,
    to_pos_inf = 2,
    to_neg_inf = 4,
    to_zero = 8,
    current_dir = 16,
    no_exc = 32,
};
consteval rounding_flags operator|(
    rounding_flags lhs, rounding_flags rhs) noexcept {
    return static_cast<rounding_flags>(
        __DPL to_underlying(lhs) | __DPL to_underlying(rhs));
}

consteval rounding_flags operator&(
    rounding_flags lhs, rounding_flags rhs) noexcept {
    return static_cast<rounding_flags>(
        __DPL to_underlying(lhs) & __DPL to_underlying(rhs));
}

consteval bool is_valid(rounding_flags val) noexcept {
    constexpr rounding_flags exclusives = rounding_flags::to_nearest_int |
        rounding_flags::to_pos_inf | rounding_flags::to_neg_inf |
        rounding_flags::to_zero | rounding_flags::current_dir;
    return __DPL popcount(
               __DPL to_unsigned(__DPL to_underlying(exclusives & val))) == 1;
}

template <rounding_flags O>
struct rounding_t {
    explicit consteval rounding_t() noexcept = default;

    template <rounding_flags O2>
    friend consteval rounding_t<O | O2> operator|(
        rounding_t, rounding_t<O2>) noexcept {
        return rounding_t<O | O2>{};
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    explicit constexpr operator bool(this rounding_t) noexcept {
        return is_valid(O);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool has(this rounding_t, rounding_flags flag) noexcept {
        return (O & flag) == flag;
    }

    template <rounding_flags O2>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr bool has(this rounding_t, rounding_t<O2>) noexcept {
        return (O & O2) == O2;
    }
};

template <rounding_flags R>
inline constexpr rounding_t<R> rounding_v{};

} // namespace datapar::fmath
__DPL_DEFAULT_NAMESPACE_END
