// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/constants/min_value.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT
struct mantissa_bits_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr mantissa_bits_t() noexcept = default;

    template <floating_point T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this mantissa_bits_t) noexcept {
        using bit_type = bit_type_t<sizeof(T) * char_bit_v>;
        auto const rep = static_cast<bit_type>(
            __DPL bit_cast<bit_type>(min_value_v<T>) - 1);
        return __DPL bit_cast<T>(rep);
    }
};

DPL_EXPORT
inline constexpr mantissa_bits_t mantissa_bits{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<mantissa_bits_t, T>
inline constexpr auto mantissa_bits_v = static_cast<T>(mantissa_bits);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
