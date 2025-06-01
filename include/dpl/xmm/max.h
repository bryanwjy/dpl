// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h" // IWYU pragma: keep

// IWYU pragma: begin_exports
#include "dpl/xmm/fwd.h"

#include "dpl/xmm/vector.h"
#include "dpl/xmm/vector_mask.h"
// IWYU pragma: end_exports

#include "dpl/xmm/concepts.h"
#include "dpl/xmm/constants.h" // IWYU pragma: keep

#if !DPL_MODULES
#  include <immintrin.h>

#  include <concepts>
#endif

DPL_NAMESPACE_BEGIN

DPL_SIMD_ABI_NAMESPACE_BEGIN(xmm)

DPL_EXPORT template <xmm_element T>
requires std::floating_point<T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> max(
    vector<T> left, vector<T> right) noexcept {
    if constexpr (sizeof(T) == 4) {
        return _mm_max_ps(+left, +right);
    } else {
        static_assert(sizeof(T) == 8, "unsupported operation");
        return _mm_max_pd(+left, +right);
    }
}

DPL_EXPORT template <xmm_scalar_element T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar<T> max(
    scalar<T> left, scalar<T> right) noexcept {
    if constexpr (sized_floating_point<T, 4>) {
        return _mm_max_ss(+left, +right);
    } else {
        static_assert(sized_floating_point<T, 8>, "unsupported operation");
        return _mm_max_sd(+left, +right);
    }
}

#if DPL_SIMD_X86_AVX512FP16 & DPL_SUPPORTS_FLOAT16
DPL_EXPORT template <xmm_element T>
requires sized_floating_point<T, 2> && std::same_as<float16, T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> max(
    vector<T> left, vector<T> right) noexcept {
    return _mm_max_ph(+left, +right);
}

DPL_EXPORT template <xmm_scalar_element T>
requires sized_floating_point<T, 2> && std::same_as<float16, T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline scalar<T> max(
    scalar<T> left, scalar<T> right) noexcept {
    return _mm_max_sh(+left, +right);
}
#endif

DPL_EXPORT template <xmm_element T>
requires std::signed_integral<T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> max(
    vector<T> left, vector<T> right) noexcept {
    if constexpr (sizeof(T) == 1) {
        return _mm_max_epi8(+left, +right);
    } else if constexpr (sizeof(T) == 2) {
        return _mm_max_epi16(+left, +right);
    } else if constexpr (sizeof(T) == 4) {
        return _mm_max_epi32(+left, +right);
    } else {
        static_assert(sizeof(T) == 8, "unsupported operation");
#if DPL_SIMD_X86_AVX512 & DPL_SIMD_X86_AVX512DQ
        return _mm_max_epi64(+left, +right);
#else
        return select(left > right, left, right);
#endif
    }
}

DPL_EXPORT template <xmm_element T>
requires std::unsigned_integral<T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, VECTORCALL, CONST, NODISCARD) inline vector<T> max(
    vector<T> left, vector<T> right) noexcept {
    if constexpr (sizeof(T) == 1) {
        return _mm_max_epu8(+left, +right);
    } else if constexpr (sizeof(T) == 2) {
        return _mm_max_epu16(+left, +right);
    } else if constexpr (sizeof(T) == 4) {
        return _mm_max_epu32(+left, +right);
    } else {
        static_assert(sizeof(T) == 8, "unsupported operation");
#if DPL_SIMD_X86_AVX512 & DPL_SIMD_X86_AVX512DQ
        return _mm_max_epu64(+left, +right);
#else
        auto const sign = vector{msb<T>};
        auto const lhs = reinterpret_as<signed_type>(left) - sign;
        auto const rhs = reinterpret_as<signed_type>(right) - sign;
        return select(lhs > rhs, left, right);
#endif
    }
}

DPL_SIMD_ABI_NAMESPACE_END(xmm)

DPL_NAMESPACE_END