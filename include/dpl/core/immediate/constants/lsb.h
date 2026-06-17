// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT struct lsb_t : broadcastable_base<lsb_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr lsb_t() noexcept = default;

    template <integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this lsb_t) noexcept {
        return static_cast<T>(1);
    }

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this lsb_t self) noexcept {
        if constexpr (sizeof(T) == sizeof(int)) {
            return __DPL bit_cast<T>(static_cast<int>(self));
        } else if constexpr (sizeof(T) == sizeof(long)) {
            return __DPL bit_cast<T>(static_cast<long>(self));
        } else if constexpr (sizeof(T) == sizeof(long long)) {
            return __DPL bit_cast<T>(static_cast<long long>(self));
        } else {
            static_assert(sizeof(T) == sizeof(short));
            return __DPL bit_cast<T>(static_cast<short>(self));
        }
    }
};

DPL_EXPORT inline constexpr lsb_t lsb{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<lsb_t, T>
inline constexpr auto lsb_v = static_cast<T>(lsb);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
