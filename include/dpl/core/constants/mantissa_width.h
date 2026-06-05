// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/constants/digits.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <floating_point T>
struct mantissa_width_t :
    integral_constant<int, digits_v<T> - 1>,
    broadcastable_base<mantissa_width_t<T>> {
    __DPL_HIDE_FROM_ABI explicit constexpr mantissa_width_t() noexcept =
        default;
};

DPL_EXPORT template <floating_point T>
inline constexpr mantissa_width_t<T> mantissa_width{};

DPL_EXPORT template <floating_point T>
inline constexpr int mantissa_width_v = mantissa_width<T>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
