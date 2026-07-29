// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/type_traits/unsigned_integral_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

struct value_bits_t : broadcastable_base<value_bits_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr value_bits_t() noexcept = default;

    template <signed_integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this value_bits_t) noexcept {
        using uint_t = unsigned_integral_type_t<__DPL type_bit_v<T>>;
        return __DPL bit_cast<T>(
            static_cast<uint_t>(static_cast<uint_t>(-1) >> 1));
    }

    template <unsigned_integral T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this value_bits_t) noexcept {
        return static_cast<T>(-1);
    }

    template <floating_point_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this value_bits_t) noexcept {
        return __DPL bit_cast<T>(~floating_point_traits<T>::signbit);
    }
};

inline constexpr value_bits_t value_bits{};

template <typename T>
requires explicitly_convertible_to<value_bits_t, T>
inline constexpr auto value_bits_v = static_cast<T>(value_bits);

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
