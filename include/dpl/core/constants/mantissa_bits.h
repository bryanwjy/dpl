// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/convertible_to.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT
struct mantissa_bits_t : broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr mantissa_bits_t() noexcept = default;

    template <common_float_with<float> T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this mantissa_bits_t) noexcept {
        return __DPL bit_cast<T>((1 << 23) - 1);
    }

    template <common_float_with<double> T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    operator T(this mantissa_bits_t) noexcept {
        return __DPL bit_cast<T>((static_cast<uint64>(1) << 52) - 1);
    }

#if DPL_SUPPORTS_FLOAT16
    template <common_float_with<float16> T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this mantissa_bits_t) noexcept {
        return __DPL bit_cast<T>(static_cast<uint16>(1 << 10) - 1);
    }
#endif

    template <brain_float T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    constexpr operator T(this mantissa_bits_t) noexcept {
        return __DPL bit_cast<T>(static_cast<uint16>(1 << 7) - 1);
    }
};

DPL_EXPORT
inline constexpr mantissa_bits_t mantissa_bits{};

DPL_EXPORT template <typename T>
requires explicitly_convertible_to<mantissa_bits_t, T>
inline constexpr auto mantissa_bits_v = static_cast<T>(mantissa_bits);

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
