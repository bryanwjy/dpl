// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/type_traits/constants.h"
#  include "dpl/std/utility/bitset.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename>
struct floating_point_traits;

DPL_EXPORT namespace details::numbers {
template <size_t N>
struct xfp {
    static_assert(80 % __DPL char_bit_v == 0);
    static constexpr auto bytes = 80zu / __DPL char_bit_v;
    struct val_t {
        unsigned char data[bytes];
    } val;
    struct padding_t {
        unsigned char data[N - bytes];
    } padding;
    consteval bitset<80> to_bitset() const {
        xfp copy{};
        copy.val = this->val;
        return __DPL bit_cast<bitset<80>>(copy);
    }
};

template <typename T>
class extended_floating_point {};

template <typename T>
concept floating_point_like =
    (floating_point<T> ||
        derived_from<remove_cv_t<T>,
            extended_floating_point<remove_cv_t<T>>>) &&
    requires { typename floating_point_traits<T>::type; };

template <typename T>
concept binary_layout_floating_point = floating_point_like<T> && requires {
    floating_point_traits<T>::width;
    floating_point_traits<T>::digits;
    floating_point_traits<T>::signbit;
    floating_point_traits<T>::mantissa_mask;
    floating_point_traits<T>::exponent_mask;
    floating_point_traits<T>::exponent_bias;
    floating_point_traits<T>::has_hidden_bit;
    floating_point_traits<T>::has_signaling_nan;
    floating_point_traits<T>::has_denormal;
    floating_point_traits<T>::has_infinity;
    floating_point_traits<T>::radix;
    requires (floating_point_traits<T>::radix == 2);
    typename size_constant<floating_point_traits<T>::width>;
    typename size_constant<floating_point_traits<T>::exponent_bias>;
    typename size_constant<floating_point_traits<T>::digits>;
    typename integral_constant<bitset<floating_point_traits<T>::width>,
        floating_point_traits<T>::signbit>;
    typename integral_constant<bitset<floating_point_traits<T>::width>,
        floating_point_traits<T>::mantissa_mask>;
    typename integral_constant<bitset<floating_point_traits<T>::width>,
        floating_point_traits<T>::exponent_mask>;
    typename bool_constant<floating_point_traits<T>::has_hidden_bit>;
    typename bool_constant<floating_point_traits<T>::has_signaling_nan>;
    typename bool_constant<floating_point_traits<T>::has_quiet_nan>;
    typename bool_constant<floating_point_traits<T>::has_denormal>;
    typename bool_constant<floating_point_traits<T>::has_infinity>;
};

#if !DPL_SUPPORTS_FLOAT16 | !DPL_SUPPORTS_BFLOAT16
// Template to prevent common base between float16 and bfloat16
template <typename T, typename V>
struct alignas(uint16) storage16 {
    static_assert(sizeof(V) == sizeof(uint16) && alignof(V) == alignof(V));
    V value;
};

#  define __DPL_FP_SELF_ARITHMETIC(OP)                                     \
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

#  define __DPL_FP_SELF_CMP(OP)                                            \
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

#  undef __DPL_FP_SELF_CMP
#  undef __DPL_FP_SELF_ARITHMETIC

#  define __DPL_FP_OPERATOR_L(RET, LT, OP)                       \
      DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)           \
      friend constexpr RET operator OP(LT lhs, T rhs) noexcept { \
          return lhs OP static_cast<LT>(rhs);                    \
      }

#  define __DPL_FP_OPERATOR_R(RET, RT, OP)                       \
      DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)           \
      friend constexpr RET operator OP(T lhs, RT rhs) noexcept { \
          return static_cast<RT>(lhs) OP rhs;                    \
      }

#  define __DPL_FP_ASSIGN(TP, OP)                                       \
      DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)                  \
      friend constexpr T& operator OP## = (T & lhs, TP rhs) noexcept {  \
          return lhs = lhs OP static_cast<T>(rhs);                      \
      }                                                                 \
      DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)                  \
      friend constexpr TP& operator OP## = (TP & lhs, T rhs) noexcept { \
          return lhs = lhs OP static_cast<TP>(rhs);                     \
      }

#  define __DPL_FP_ARITHMETIC(TP, OP) \
      __DPL_FP_OPERATOR_L(TP, TP, OP) \
      __DPL_FP_OPERATOR_R(TP, TP, OP) \
      __DPL_FP_ASSIGN(TP, OP)         \
      static_assert(true)

#  define __DPL_FP_CMP(TP, OP)          \
      __DPL_FP_OPERATOR_L(bool, TP, OP) \
      __DPL_FP_OPERATOR_R(bool, TP, OP) \
      static_assert(true)

#  define __DPL_FP_SS(TP)                \
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
#  undef __DPL_FP_SS
#  undef __DPL_FP_CMP
#  undef __DPL_FP_ARITHMETIC
#  undef __DPL_FP_ASSIGN
#  undef __DPL_FP_OPERATOR_R
#  undef __DPL_FP_OPERATOR_L
#endif

template <typename T>
consteval T invalid_number() noexcept {
    DPL_BUILTIN_unreachable();
}

} // namespace details::numbers
DPL_DEFAULT_NAMESPACE_END
