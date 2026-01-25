// Copyright 2025 Bryan Wong

#pragma once

#if !defined(DPL_MODULES) || !DPL_MODULES
#  include "dpl/config.h"

#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace details {
template <typename T>
consteval T invalid_number() noexcept {
    DPL_BUILTIN_unreachable();
}
} // namespace details
DPL_DEFAULT_NAMESPACE_END

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT
namespace numbers {
template <typename T>
inline constexpr T e_v = details::invalid_number<T>();
template <typename T>
inline constexpr T log2e_v = details::invalid_number<T>();
template <typename T>
inline constexpr T log10e_v = details::invalid_number<T>();
template <typename T>
inline constexpr T rad_per_deg_v = details::invalid_number<T>();
template <typename T>
inline constexpr T deg_per_rad_v = details::invalid_number<T>();
template <typename T>
inline constexpr T pi_v = details::invalid_number<T>();
template <typename T>
inline constexpr T half_pi_v = details::invalid_number<T>();
template <typename T>
inline constexpr T inv_pi_v = details::invalid_number<T>();
template <typename T>
inline constexpr T inv_sqrtpi_v = details::invalid_number<T>();
template <typename T>
inline constexpr T ln2_v = details::invalid_number<T>();
template <typename T>
inline constexpr T ln10_v = details::invalid_number<T>();
template <typename T>
inline constexpr T sqrt2_v = details::invalid_number<T>();
template <typename T>
inline constexpr T sqrt3_v = details::invalid_number<T>();
template <typename T>
inline constexpr T inv_sqrt3_v = details::invalid_number<T>();
template <typename T>
inline constexpr T egamma_v = details::invalid_number<T>();
template <typename T>
inline constexpr T phi_v = details::invalid_number<T>();
} // namespace numbers

DPL_EXPORT
namespace numbers {
template <floating_point T>
inline constexpr T e_v<T> = 2.718281828459045235360287471352662;
template <floating_point T>
inline constexpr T log2e_v<T> = 1.442695040888963407359924681001892;
template <floating_point T>
inline constexpr T log10e_v<T> = 0.434294481903251827651128918916605;
template <floating_point T>
inline constexpr T pi_v<T> = 3.141592653589793238462643383279502;
template <floating_point T>
inline constexpr T rad_per_deg_v<T> = 0.0174532925199432957692369076848861;
template <floating_point T>
inline constexpr T deg_per_rad_v<T> = 57.29577951308232087679815481410517;
template <floating_point T>
inline constexpr T half_pi_v<T> = 1.570796326794896619231321691639751;
template <floating_point T>
inline constexpr T inv_pi_v<T> = 0.318309886183790671537767526745028;
template <floating_point T>
inline constexpr T inv_sqrtpi_v<T> = 0.564189583547756286948079451560772;
template <floating_point T>
inline constexpr T ln2_v<T> = 0.693147180559945309417232121458176;
template <floating_point T>
inline constexpr T ln10_v<T> = 2.302585092994045684017991454684364;
template <floating_point T>
inline constexpr T sqrt2_v<T> = 1.414213562373095048801688724209698;
template <floating_point T>
inline constexpr T sqrt3_v<T> = 1.732050807568877293527446341505872;
template <floating_point T>
inline constexpr T inv_sqrt3_v<T> = 0.577350269189625764509148780501957;

template <floating_point T>
inline constexpr T egamma_v<T> = 0.577215664901532860606512090082402;
template <floating_point T>
inline constexpr T phi_v<T> = 1.618033988749894848204586834365638;

} // namespace numbers

DPL_DEFAULT_NAMESPACE_END
