#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include <cfloat>
#  include <climits>
#  include <concepts>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

/**
 * Non-standard limits for internal use
 */
namespace numeric {
static_assert((-1 & 3) == 3, "Invalid signed representation detected");

template <typename T>
inline constexpr T minimum_v;

template <typename T>
inline constexpr T lowest_v;

template <typename T>
inline constexpr T maximum_v;

template <typename T>
inline constexpr T signbit_v;

template <typename T>
inline constexpr T all_bits_v;

template <std::signed_integral T>
inline constexpr T minimum_v<T> =
    static_cast<T>(static_cast<T>(1) << (CHAR_BIT * sizeof(T) - 1));

template <std::unsigned_integral T>
inline constexpr T minimum_v<T> = 0;

template <std::integral T>
inline constexpr T maximum_v<T> = static_cast<T>(~minimum_v<T>);

template <std::integral T>
inline constexpr T all_bits_v = ~static_cast<T>(0);

template <>
inline constexpr bool all_bits_v<bool> = true;

template <std::floating_point T>
inline constexpr T all_bits_v<T> = []() {
    signed char data[sizeof(T)]{};
    for (auto& val : data) {
        val = ~val;
    }
    return DPL_BUILTIN_bit_cast(T, data);
}();

template <>
inline constexpr bool maximum_v<bool> = true;

template <>
inline constexpr bool minimum_v<bool> = false;

template <>
inline constexpr float maximum_v<float> = FLT_MAX;

template <>
inline constexpr float minimum_v<float> = FLT_MIN;

template <>
inline constexpr double maximum_v<double> = DBL_MAX;

template <>
inline constexpr double minimum_v<double> = DBL_MIN;

template <>
inline constexpr long double maximum_v<long double> = LDBL_MAX;

template <>
inline constexpr long double minimum_v<long double> = LDBL_MIN;

template <std::integral T>
inline constexpr T lowest_v<T> = minimum_v<T>;

template <std::floating_point T>
inline constexpr T lowest_v<T> = -maximum_v<T>;

template <std::floating_point T>
inline constexpr T signbit_v<T> = -0.0;

template <std::signed_integral T>
inline constexpr T signbit_v<T> = minimum_v<T>;

// TODO disable -Wno-undef for region
#if DPL_SUPPORTS_FLOAT16
template <>
inline constexpr float16 maximum_v<float16> = 6.5504e+4f16;

template <>
inline constexpr float16 minimum_v<float16> = 6.103515625e-5f16;

#endif
#if DPL_SUPPORTS_FLOAT32
template <>
inline constexpr float32 maximum_v<float32> = 3.40282347e+38f32;

template <>
inline constexpr float32 minimum_v<float32> = 1.17549435e-38f32;

#endif
#if DPL_SUPPORTS_FLOAT64
template <>
inline constexpr float64 maximum_v<float64> = 1.7976931348623157e+308f64;

template <>
inline constexpr float64 minimum_v<float64> = 2.2250738585072014e-308f64;

#endif
#if DPL_SUPPORTS_FLOAT128
template <>
inline constexpr float128 maximum_v<float128> =
    1.1897314953572317650857593266280070162e+4932f128;

template <>
inline constexpr float128 minimum_v<float128> =
    3.3621031431120935062626778173217526026e-4932f128;
#endif
#if DPL_SUPPORTS_BFLOAT16
template <>
inline constexpr bfloat16 maximum_v<bfloat16> = 3.38953139e+38bf16;

template <>
inline constexpr bfloat16 minimum_v<bfloat16> = 1.175494351e-38bf16;
#endif
} // namespace numeric

DPL_DEFAULT_NAMESPACE_END