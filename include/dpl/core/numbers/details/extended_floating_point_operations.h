// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/numbers/details/extended_floating_point.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::numbers {
#define __DPL_FP_SELF_ARITHMETIC(OP)                                     \
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)                     \
    friend constexpr T operator OP(                                      \
        extended_floating_point_operations const& lhs, T rhs) noexcept { \
        return static_cast<T>(lhs) OP rhs;                               \
    }                                                                    \
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)                     \
    friend constexpr T& operator OP## =                                  \
        (extended_floating_point_operations & lhs, T rhs) noexcept {     \
        return static_cast<T&>(lhs) = static_cast<T const&>(lhs) OP rhs; \
    }                                                                    \
    static_assert(true)

#define __DPL_FP_SELF_CMP(OP)                                            \
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)                     \
    friend constexpr bool operator OP(                                   \
        extended_floating_point_operations const& lhs, T rhs) noexcept { \
        return static_cast<T>(lhs) OP rhs;                               \
    }                                                                    \
    static_assert(true)

template <typename T>
struct extended_floating_point_operations : public extended_floating_point<T> {
    __DPL_FP_SELF_ARITHMETIC(+);
    __DPL_FP_SELF_ARITHMETIC(-);
    __DPL_FP_SELF_ARITHMETIC(*);
    __DPL_FP_SELF_ARITHMETIC(/);

    __DPL_FP_SELF_CMP(<);
    __DPL_FP_SELF_CMP(>);
    __DPL_FP_SELF_CMP(<=);
    __DPL_FP_SELF_CMP(>=);
    __DPL_FP_SELF_CMP(==);
    __DPL_FP_SELF_CMP(!=);

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    friend constexpr auto operator<=>(
        extended_floating_point_operations const& lhs, T rhs) noexcept {
        return static_cast<T>(lhs) <=> rhs;
    }
};

#undef __DPL_FP_SELF_CMP
#undef __DPL_FP_SELF_ARITHMETIC

} // namespace details::numbers
__DPL_DEFAULT_NAMESPACE_END
