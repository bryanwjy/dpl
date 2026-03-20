// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/constants/one.h"

#if !DPL_MODULES
#  include "dpl/core/basic/broadcastable_base.h"
#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/countr.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
template <arithmetic_type T>
consteval int digits_of() noexcept {
    if constexpr (signed_integral<T>) {
        return sizeof(T) * char_bit_v - 1;
    } else if constexpr (unsigned_integral<T>) {
        return sizeof(T) * char_bit_v;
    } else {
        static_assert(floating_point<T>);
        using bit_type = bit_type_t<char_bit_v * sizeof(T)>;
        return __DPL countr_zero( __DPL bit_cast<bit_type>(one_v<T>)) + 1;
    }
}

DPL_EXPORT template <arithmetic_type T>
struct digits_t : integral_constant<int, digits_of<T>()>, broadcastable_base {
    __DPL_HIDE_FROM_ABI explicit constexpr digits_t() noexcept = default;
};

DPL_EXPORT template <arithmetic_type T>
inline constexpr digits_t<T> digits{};

DPL_EXPORT template <arithmetic_type T>
inline constexpr int digits_v = digits<T>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
