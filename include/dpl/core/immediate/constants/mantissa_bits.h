// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"
#include "dpl/core/immediate/constants/min_value.h"

#if !DPL_MODULES
#  include "dpl/core/numbers/floating_point_traits.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/convertible_to.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

struct mantissa_bits_t : broadcastable_base<mantissa_bits_t> {
    __DPL_HIDE_FROM_ABI explicit constexpr mantissa_bits_t() noexcept = default;

    template <floating_point_like T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this mantissa_bits_t) noexcept {
        return __DPL bit_cast<T>(floating_point_traits<T>::mantissa_mask);
    }
};

inline constexpr mantissa_bits_t mantissa_bits{};

template <typename T>
requires explicitly_convertible_to<mantissa_bits_t, T>
inline constexpr auto mantissa_bits_v = static_cast<T>(mantissa_bits);

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
