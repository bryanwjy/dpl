// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT namespace details::numbers {

#define __DPL_FP_OPERATOR_L(RET, LT, OP)                       \
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)           \
    friend constexpr RET operator OP(LT lhs, T rhs) noexcept { \
        return lhs OP static_cast<LT>(rhs);                    \
    }

#define __DPL_FP_OPERATOR_R(RET, RT, OP)                       \
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)           \
    friend constexpr RET operator OP(T lhs, RT rhs) noexcept { \
        return static_cast<RT>(lhs) OP rhs;                    \
    }

#define __DPL_FP_ASSIGN(TP, OP)                                       \
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)                  \
    friend constexpr T& operator OP## = (T & lhs, TP rhs) noexcept {  \
        return lhs = lhs OP static_cast<T>(rhs);                      \
    }                                                                 \
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)                  \
    friend constexpr TP& operator OP## = (TP & lhs, T rhs) noexcept { \
        return lhs = lhs OP static_cast<TP>(rhs);                     \
    }

#define __DPL_FP_ARITHMETIC(TP, OP) \
    __DPL_FP_OPERATOR_L(TP, TP, OP) \
    __DPL_FP_OPERATOR_R(TP, TP, OP) \
    __DPL_FP_ASSIGN(TP, OP)         \
    static_assert(true)

#define __DPL_FP_CMP(TP, OP)          \
    __DPL_FP_OPERATOR_L(bool, TP, OP) \
    __DPL_FP_OPERATOR_R(bool, TP, OP) \
    static_assert(true)

#define __DPL_FP_SS(TP)                \
    __DPL_FP_OPERATOR_L(auto, TP, <=>) \
    __DPL_FP_OPERATOR_R(auto, TP, <=>) \
    static_assert(true)

template <typename T, typename To>
struct promotable {
    __DPL_FP_ARITHMETIC(To, +);
    __DPL_FP_ARITHMETIC(To, -);
    __DPL_FP_ARITHMETIC(To, *);
    __DPL_FP_ARITHMETIC(To, /);
    __DPL_FP_CMP(To, <);
    __DPL_FP_CMP(To, >);
    __DPL_FP_CMP(To, <=);
    __DPL_FP_CMP(To, >=);
    __DPL_FP_CMP(To, ==);
    __DPL_FP_CMP(To, !=);
    __DPL_FP_SS(To);
};
#undef __DPL_FP_SS
#undef __DPL_FP_CMP
#undef __DPL_FP_ARITHMETIC
#undef __DPL_FP_ASSIGN
#undef __DPL_FP_OPERATOR_R
#undef __DPL_FP_OPERATOR_L

} // namespace details::numbers
DPL_DEFAULT_NAMESPACE_END
