// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"

#if !DPL_MODULES
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/type_traits/unsigned_integral_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

struct msb_t : broadcastable_base<msb_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr msb_t() noexcept = default;

    template <integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this msb_t) noexcept {
        constexpr auto width = __DPL type_bit_v<T>;
        using uint_t = unsigned_integral_type_t<width>;
        auto const result = static_cast<uint_t>(1) << (width - 1);
        return static_cast<T>(result);
    }

    template <floating_point_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this msb_t) noexcept {
        return __DPL bit_cast<T>(floating_point_traits<T>::signbit);
    }
};

inline constexpr msb_t msb{};

template <typename T>
requires explicitly_convertible_to<msb_t, T>
inline constexpr auto msb_v = static_cast<T>(msb);

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
