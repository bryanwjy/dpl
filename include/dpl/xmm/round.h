// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

#include "dpl/xmm/scalar.h"
#include "dpl/xmm/vector.h"

#if !DPL_MODULES
#  include <immintrin.h>
#endif

DPL_EXPORTED_DEFAULT_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

enum class rounding : int {
    nearest_int = _MM_FROUND_TO_NEAREST_INT,
    neg_inf = _MM_FROUND_TO_NEG_INF,
    pos_inf = _MM_FROUND_TO_POS_INF,
    zero = _MM_FROUND_TO_ZERO,
    noexc = _MM_FROUND_NO_EXC,
};

consteval rounding operator|(rounding left, rounding right) noexcept {
    return static_cast<rounding>(
        static_cast<int>(left) | static_cast<int>(right));
}

consteval rounding& operator|=(rounding& left, rounding right) noexcept {
    return left = (left | right);
}

template <rounding R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) vector<float> round(
    vector<float> val) noexcept {
    return _mm_round_ps(+va, R);
}

template <rounding R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) scalar<float> round(
    scalar<float> val) noexcept {
    return _mm_round_ss(+va, R);
}

template <rounding R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) vector<double> round(
    vector<double> val) noexcept {
    return _mm_round_pd(+va, R);
}

template <rounding R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) scalar<double> round(
    scalar<double> val) noexcept {
    return _mm_round_sd(+va, R);
}

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_EXPORTED_DEFAULT_NAMESPACE_END
