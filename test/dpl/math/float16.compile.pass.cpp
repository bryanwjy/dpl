// Copyright 2026 Bryan Wong
#include "dpl/config.h"

#include <cassert>

import dpl;

dpl::ext::bfloat16 ambiguity_check(dpl::ext::bfloat16) noexcept;
#if DPL_SUPPORTS_FLOAT32
dpl::float32 ambiguity_check(dpl::float32) noexcept;
#endif
#if DPL_SUPPORTS_FLOAT64
dpl::float64 ambiguity_check(dpl::float64) noexcept;
#endif
#if DPL_SUPPORTS_FLOAT128
dpl::float128 ambiguity_check(dpl::float128) noexcept;
#endif
float ambiguity_check(float) noexcept;
double ambiguity_check(double) noexcept;
long double ambiguity_check(long double) noexcept;
template <typename T>
concept ambiguous = !requires(T val) { ambiguity_check(val); } && requires {
    requires dpl::convertible_to<T, float> && dpl::convertible_to<T, double> &&
        dpl::convertible_to<T, long double>;
    requires dpl::convertible_to<T, dpl::ext::bfloat16>;
#if DPL_SUPPORTS_FLOAT32
    requires dpl::convertible_to<T, dpl::float32>;
#endif
#if DPL_SUPPORTS_FLOAT64
    requires dpl::convertible_to<T, dpl::float64>;
#endif
#if DPL_SUPPORTS_FLOAT128
    requires dpl::convertible_to<T, dpl::float128>;
#endif
};

static_assert(sizeof(dpl::ext::float16) == sizeof(dpl::int16) &&
    alignof(dpl::ext::float16) == alignof(dpl::int16));
static_assert(ambiguous<dpl::ext::float16>);
static_assert(dpl::regular<dpl::ext::float16>);
static_assert(dpl::totally_ordered<dpl::ext::float16>);
static_assert(dpl::is_standard_layout_v<dpl::ext::float16>);
static_assert(dpl::is_trivially_copyable_v<dpl::ext::float16>);
static_assert(
    dpl::explicitly_convertible_to<dpl::ext::float16, dpl::ext::bfloat16>);
static_assert(
    dpl::explicitly_convertible_to<dpl::ext::bfloat16, dpl::ext::float16>);

static_assert(dpl::same_as<dpl::ext::float16, decltype(+dpl::ext::float16{})>);
static_assert(dpl::same_as<dpl::ext::float16, decltype(-dpl::ext::float16{})>);
static_assert(dpl::same_as<dpl::ext::float16,
    decltype(dpl::ext::float16{} + dpl::ext::float16{})>);
static_assert(dpl::same_as<dpl::ext::float16,
    decltype(dpl::ext::float16{} - dpl::ext::float16{})>);
static_assert(dpl::same_as<dpl::ext::float16,
    decltype(dpl::ext::float16{} * dpl::ext::float16{})>);
static_assert(dpl::same_as<dpl::ext::float16,
    decltype(dpl::ext::float16{} / dpl::ext::float16{})>);
DPL_DISABLE_WARNING_PUSH()
#if DPL_COMPILER_CLANG
DPL_DISABLE_WARNING("-Wunevaluated-expression")
#endif
static_assert(dpl::same_as<dpl::ext::float16&,
    decltype(dpl::declval<dpl::ext::float16&>() += dpl::ext::float16{})>);
static_assert(dpl::same_as<dpl::ext::float16&,
    decltype(dpl::declval<dpl::ext::float16&>() -= dpl::ext::float16{})>);
static_assert(dpl::same_as<dpl::ext::float16&,
    decltype(dpl::declval<dpl::ext::float16&>() *= dpl::ext::float16{})>);
static_assert(dpl::same_as<dpl::ext::float16&,
    decltype(dpl::declval<dpl::ext::float16&>() /= dpl::ext::float16{})>);
DPL_DISABLE_WARNING_POP()
#define __DPL_DEFINE_PROMOTION_TEST(TP, OP)                     \
    static_assert(dpl::same_as<TP,                              \
        decltype(dpl::ext::float16 {} OP static_cast<TP>(1))>); \
    static_assert(dpl::same_as<TP,                              \
        decltype(static_cast<TP>(1) OP dpl::ext::float16{})>);  \
    static_assert(dpl::same_as<dpl::ext::float16&,              \
        decltype(dpl::declval<dpl::ext::float16&>() OP## =      \
                     static_cast<TP>(1))>);                     \
    static_assert(dpl::same_as<TP&,                             \
        decltype(dpl::declval<TP&>() OP## = dpl::ext::float16{1})>);

__DPL_DEFINE_PROMOTION_TEST(float, +);
__DPL_DEFINE_PROMOTION_TEST(float, -);
__DPL_DEFINE_PROMOTION_TEST(float, *);
__DPL_DEFINE_PROMOTION_TEST(float, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, float>);
static_assert(dpl::explicitly_convertible_to<float, dpl::ext::float16>);
static_assert(dpl::convertible_to<dpl::ext::float16, float>);
static_assert(dpl::common_with<dpl::ext::float16, float>);
static_assert(
    dpl::same_as<dpl::common_type_t<dpl::ext::float16, float>, float>);
__DPL_DEFINE_PROMOTION_TEST(double, +);
__DPL_DEFINE_PROMOTION_TEST(double, -);
__DPL_DEFINE_PROMOTION_TEST(double, *);
__DPL_DEFINE_PROMOTION_TEST(double, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, double>);
static_assert(dpl::explicitly_convertible_to<double, dpl::ext::float16>);
static_assert(dpl::convertible_to<dpl::ext::float16, double>);
static_assert(dpl::common_with<dpl::ext::float16, double>);
static_assert(
    dpl::same_as<dpl::common_type_t<dpl::ext::float16, double>, double>);
__DPL_DEFINE_PROMOTION_TEST(long double, +);
__DPL_DEFINE_PROMOTION_TEST(long double, -);
__DPL_DEFINE_PROMOTION_TEST(long double, *);
__DPL_DEFINE_PROMOTION_TEST(long double, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, long double>);
static_assert(dpl::explicitly_convertible_to<long double, dpl::ext::float16>);
static_assert(dpl::convertible_to<dpl::ext::float16, long double>);
static_assert(dpl::common_with<dpl::ext::float16, long double>);
static_assert(dpl::same_as<dpl::common_type_t<dpl::ext::float16, long double>,
    long double>);

#if DPL_SUPPORTS_FLOAT32
__DPL_DEFINE_PROMOTION_TEST(dpl::float32, +);
__DPL_DEFINE_PROMOTION_TEST(dpl::float32, -);
__DPL_DEFINE_PROMOTION_TEST(dpl::float32, *);
__DPL_DEFINE_PROMOTION_TEST(dpl::float32, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, dpl::float32>);
static_assert(dpl::explicitly_convertible_to<dpl::float32, dpl::ext::float16>);
static_assert(dpl::convertible_to<dpl::ext::float16, dpl::float32>);
static_assert(dpl::common_with<dpl::ext::float16, dpl::float32>);
static_assert(dpl::same_as<dpl::common_type_t<dpl::ext::float16, dpl::float32>,
    dpl::float32>);
#endif
#if DPL_SUPPORTS_FLOAT64
__DPL_DEFINE_PROMOTION_TEST(dpl::float64, +);
__DPL_DEFINE_PROMOTION_TEST(dpl::float64, -);
__DPL_DEFINE_PROMOTION_TEST(dpl::float64, *);
__DPL_DEFINE_PROMOTION_TEST(dpl::float64, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, dpl::float64>);
static_assert(dpl::explicitly_convertible_to<dpl::float64, dpl::ext::float16>);
static_assert(dpl::convertible_to<dpl::ext::float16, dpl::float64>);
static_assert(dpl::common_with<dpl::ext::float16, dpl::float64>);
static_assert(dpl::same_as<dpl::common_type_t<dpl::ext::float16, dpl::float64>,
    dpl::float64>);
#endif
#if DPL_SUPPORTS_FLOAT128
__DPL_DEFINE_PROMOTION_TEST(dpl::float128, +);
__DPL_DEFINE_PROMOTION_TEST(dpl::float128, -);
__DPL_DEFINE_PROMOTION_TEST(dpl::float128, *);
__DPL_DEFINE_PROMOTION_TEST(dpl::float128, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, dpl::float128>);
static_assert(dpl::explicitly_convertible_to<dpl::float128, dpl::ext::float16>);
static_assert(dpl::convertible_to<dpl::ext::float16, dpl::float128>);
static_assert(dpl::common_with<dpl::ext::float16, dpl::float128>);
static_assert(dpl::same_as<dpl::common_type_t<dpl::ext::float16, dpl::float128>,
    dpl::float128>);
#endif
#undef __DPL_DEFINE_PROMOTION_TEST

#define __DPL_DEFINE_INT_PROMOTION_TEST(TP, OP)                 \
    static_assert(dpl::same_as<dpl::ext::float16,               \
        decltype(dpl::ext::float16 {} OP static_cast<TP>(1))>); \
    static_assert(dpl::same_as<dpl::ext::float16,               \
        decltype(static_cast<TP>(1) OP dpl::ext::float16{})>);  \
    static_assert(dpl::same_as<dpl::ext::float16&,              \
        decltype(dpl::declval<dpl::ext::float16&>() OP## =      \
                     static_cast<TP>(1))>);                     \
    static_assert(dpl::same_as<TP&,                             \
        decltype(dpl::declval<TP&>() OP## = dpl::ext::float16{1})>);

__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int8, +);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int8, -);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int8, *);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int8, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, dpl::int8>);
static_assert(dpl::common_with<dpl::ext::float16, dpl::int8>);
static_assert(dpl::same_as<dpl::common_type_t<dpl::ext::float16, dpl::int8>,
    dpl::ext::float16>);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint8, +);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint8, -);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint8, *);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint8, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, dpl::uint8>);
static_assert(dpl::common_with<dpl::ext::float16, dpl::uint8>);
static_assert(dpl::same_as<dpl::common_type_t<dpl::ext::float16, dpl::uint8>,
    dpl::ext::float16>);

__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int16, +);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int16, -);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int16, *);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int16, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, dpl::int16>);
static_assert(dpl::common_with<dpl::ext::float16, dpl::int16>);
static_assert(dpl::same_as<dpl::common_type_t<dpl::ext::float16, dpl::int16>,
    dpl::ext::float16>);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint16, +);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint16, -);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint16, *);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint16, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, dpl::uint16>);
static_assert(dpl::common_with<dpl::ext::float16, dpl::uint16>);
static_assert(dpl::same_as<dpl::common_type_t<dpl::ext::float16, dpl::uint16>,
    dpl::ext::float16>);

__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int32, +);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int32, -);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int32, *);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int32, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, dpl::int32>);
static_assert(dpl::common_with<dpl::ext::float16, dpl::int32>);
static_assert(dpl::same_as<dpl::common_type_t<dpl::ext::float16, dpl::int32>,
    dpl::ext::float16>);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint32, +);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint32, -);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint32, *);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint32, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, dpl::uint32>);
static_assert(dpl::common_with<dpl::ext::float16, dpl::uint32>);
static_assert(dpl::same_as<dpl::common_type_t<dpl::ext::float16, dpl::uint32>,
    dpl::ext::float16>);

__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int64, +);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int64, -);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int64, *);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::int64, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, dpl::int64>);
static_assert(dpl::common_with<dpl::ext::float16, dpl::int64>);
static_assert(dpl::same_as<dpl::common_type_t<dpl::ext::float16, dpl::int64>,
    dpl::ext::float16>);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint64, +);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint64, -);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint64, *);
__DPL_DEFINE_INT_PROMOTION_TEST(dpl::uint64, /);
static_assert(dpl::totally_ordered_with<dpl::ext::float16, dpl::uint64>);
static_assert(dpl::common_with<dpl::ext::float16, dpl::uint64>);
static_assert(dpl::same_as<dpl::common_type_t<dpl::ext::float16, dpl::uint64>,
    dpl::ext::float16>);

#undef __DPL_DEFINE_INT_PROMOTION_TEST
