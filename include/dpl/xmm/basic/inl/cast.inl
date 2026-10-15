// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  if !DPL_MODULES
#    include "dpl/xmm/basic/abi.h"
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm::fwd {

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline __m128 castsi128_ps(
    __m128i val) noexcept {
    return _mm_castsi128_ps(val);
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline __m128d castsi128_pd(
    __m128i val) noexcept {
    return _mm_castsi128_pd(val);
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline __m128i castpd_si128(
    __m128d val) noexcept {
    return _mm_castpd_si128(val);
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline __m128i castps_si128(
    __m128 val) noexcept {
    return _mm_castps_si128(val);
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline __m128i castph_si128(
    __m128h val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16
    return _mm_castph_si128(val);
#  else
    return __DPL bit_cast<__m128i>(val);
#  endif
}
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline __m128h castsi128_ph(
    __m128i val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16
    return _mm_castsi128_ph(val);
#  else
    return __DPL bit_cast<__m128h>(val);
#  endif
}

} // namespace datapar::xmm::fwd

__DPL_DEFAULT_NAMESPACE_END

#endif
