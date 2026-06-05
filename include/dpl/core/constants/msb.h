// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT struct msb_t : broadcastable_base<msb_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr msb_t() noexcept = default;

    template <integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this msb_t) noexcept {
        constexpr auto width = char_bit_v * sizeof(T);
        auto const result = static_cast<bit_type_t<width>>(1) << (width - 1);
        return static_cast<T>(result);
    }

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this msb_t) noexcept {
        return -0.0;
    }
};

DPL_EXPORT inline constexpr msb_t msb{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<msb_t, T>
inline constexpr auto msb_v = static_cast<T>(msb);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
