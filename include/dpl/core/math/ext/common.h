// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/floating_point_traits.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/stddef/types.h"
#  include "dpl/std/type_traits/common_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::ext {
#if (!DPL_SUPPORTS_FLOAT16) | (!DPL_SUPPORTS_BFLOAT16)
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
struct extended_floating_point_operations :
    public internal::extended_floating_point<T> {
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

#if !DPL_SUPPORTS_BFLOAT16
class bfloat16_t;
DPL_EXPORT using bfloat16 = bfloat16_t;
#else
DPL_EXPORT using bfloat16 = __DPL bfloat16;
#endif
#if !DPL_SUPPORTS_FLOAT16
class float16_t;
DPL_EXPORT using float16 = float16_t;
#else
DPL_EXPORT using float16 = __DPL float16;
#endif

namespace dx = __DPL datapar; // NOLINT
inline namespace ext_literals {}
} // namespace datapar::ext

DPL_EXPORT namespace ext = datapar::ext; // NOLINT

namespace datapar {
DPL_EXPORT namespace ext_literals = ext::ext_literals;          // NOLINT
} // namespace datapar

DPL_EXPORT namespace ext_literals = datapar::ext::ext_literals; // NOLINT

#if !DPL_SUPPORTS_BFLOAT16
DPL_EXPORT template <floating_point T>
struct common_type<ext::bfloat16, T> {
    using type = T;
};
DPL_EXPORT template <floating_point T>
struct common_type<T, ext::bfloat16> {
    using type = T;
};
DPL_EXPORT template <integral T>
struct common_type<T, ext::bfloat16> {
    using type = ext::bfloat16;
};
DPL_EXPORT template <integral T>
struct common_type<ext::bfloat16, T> {
    using type = ext::bfloat16;
};

DPL_EXPORT template <>
struct floating_point_traits<ext::bfloat16> {
    using type = ext::bfloat16;

    static constexpr auto width = 16zu;
    static constexpr auto digits = 8zu;
    static constexpr auto signbit = bitset<16>(1u << 15);
    static constexpr auto mantissa_mask =
        __DPL truncate<16>(floating_point_traits<float>::mantissa_mask >> 16);
    static constexpr auto exponent_mask =
        __DPL truncate<16>(floating_point_traits<float>::exponent_mask >> 16);
    static constexpr auto has_hidden_bit = true;
};
#endif

#if !DPL_SUPPORTS_FLOAT16
DPL_EXPORT template <floating_point T>
struct common_type<ext::float16, T> {
    using type = T;
};
DPL_EXPORT template <floating_point T>
struct common_type<T, ext::float16> {
    using type = T;
};
DPL_EXPORT template <integral T>
struct common_type<T, ext::float16> {
    using type = ext::float16;
};
DPL_EXPORT template <integral T>
struct common_type<ext::float16, T> {
    using type = ext::float16;
};

DPL_EXPORT template <>
struct floating_point_traits<ext::float16> {
    using type = ext::float16;

    static constexpr auto width = 16zu;
    static constexpr auto digits = 11zu;
    static constexpr auto signbit = bitset<16>(1u << 15);
    static constexpr auto mantissa_mask = bitset<16>((1u << (digits - 1)) - 1);
    static constexpr auto exponent_mask = ~mantissa_mask ^ signbit;
    static constexpr auto has_hidden_bit = true;
};
#endif
DPL_DEFAULT_NAMESPACE_END
