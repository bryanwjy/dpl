// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/immediate/broadcastable_base.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/type_traits/floating_point_traits.h"
#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
template <integral T>
consteval int digits_of() noexcept {
    if constexpr (signed_integral<T>) {
        return sizeof(T) * char_bit_v - 1;
    } else {
        return sizeof(T) * char_bit_v;
    }
}

template <floating_point_like T>
consteval int digits_of() noexcept {
    return static_cast<int>(floating_point_traits<T>::digits);
}

DPL_EXPORT template <basic_element T>
struct digits_t :
    integral_constant<int, digits_of<T>()>,
    broadcastable_base<digits_t<T>> {
    __DPL_HIDE_FROM_ABI explicit constexpr digits_t() noexcept = default;
};

DPL_EXPORT template <basic_element T>
inline constexpr digits_t<T> digits{};

DPL_EXPORT template <basic_element T>
inline constexpr int digits_v = digits<T>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
