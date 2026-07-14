// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"
#include "dpl/core/immediate/constants/digits.h"

#if !DPL_MODULES
#  include "dpl/core/numbers/floating_point_like.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <floating_point_like T>
struct mantissa_width_t :
    integral_constant<int, digits_v<T> - 1>,
    broadcastable_base<mantissa_width_t<T>> {
    __DPL_HIDE_FROM_ABI explicit constexpr mantissa_width_t() noexcept =
        default;
};

template <floating_point_like T>
inline constexpr mantissa_width_t<T> mantissa_width{};

template <floating_point_like T>
inline constexpr int mantissa_width_v = mantissa_width<T>;

} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
