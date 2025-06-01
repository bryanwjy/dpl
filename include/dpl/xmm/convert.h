// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/xmm/bitset.h"
#include "dpl/xmm/constatns.h"
#include "dpl/xmm/vector.h"

#if !DPL_MODULES
#  include <immintrin.h>

#  include <concepts>
#endif

DPL_EXPORTED_DEFAULT_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

template <std::same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) vector<To> convert(
    vector<int> val) noexcept {
    return _mm_cvtepi32_ps(+val);
}

template <std::same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) vector<To> convert(
    vector<int> val) noexcept {
    return _mm_cvtepi32_pd(+val);
}

template <std::same_as<int> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) vector<To> convert(
    vector<double> val) noexcept {
    return _mm_cvtpd_epi32(+val);
}

template <std::same_as<int> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) vector<To> convert(
    vector<float> val) noexcept {
    return _mm_cvtps_epi32(+val);
}

template <std::same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) scalar<To> convert(
    vector<To> upper, int val) noexcept {
    return _mm_cvtsi32_ss(+upper, *val);
}

template <std::same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) vector<To> convert(
    vector<To> upper, int val) noexcept {
    return _mm_cvtsi32_ss(+upper, *val);
}

template <std::same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) scalar<To> convert(
    scalar<int> val) noexcept {
    return convert(vector{zero<float>}, *val);
}

template <std::same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) scalar<To> convert(
    scalar<int> val) noexcept {
    return convert(vector{zero<double>}, *val);
}

template <std::same_as<int> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) To convert(
    scalar<double> val) noexcept {
    return _mm_cvtsd_si32(*val);
}

template <std::same_as<int> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) To convert(
    scalar<float> val) noexcept {
    return _mm_cvtss_si32(+val);
}

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_EXPORTED_DEFAULT_NAMESPACE_END
