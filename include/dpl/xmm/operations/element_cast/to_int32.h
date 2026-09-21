// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

#  include "dpl/xmm/operations/arithmetic/to_float.h"
#  include "dpl/xmm/operations/element_cast/fallback.h" // IWYU pragma: export
#  include "dpl/xmm/operations/element_cast/to_int16.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/zero.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <same_as<int32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int64> val) noexcept {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi64_epi32(+val);
#  else
    return _mm_castps_si128(_mm_shuffle_ps(_mm_castsi128_ps(+val),
        +xmm::broadcast<float>(dx::zero), _MM_SHUFFLE(2, 0, 2, 0)));
#  endif
}

template <same_as<int32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int16> val) noexcept {
#  if DPL_SIMD_X86_SSE4_1
    return _mm_cvtepi16_epi32(+val);
#  else
    auto const neg = _mm_cmplt_epi16(+val, +xmm::broadcast<int16>(dx::zero));
    return _mm_unpacklo_epi16(+val, neg);
#  endif
}

template <same_as<int32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint16> val) noexcept {
#  if DPL_SIMD_X86_SSE4_1
    return _mm_cvtepu16_epi32(+val);
#  else
    return _mm_unpacklo_epi16(+val, +xmm::broadcast<int16>(dx::zero));
#  endif
}

template <same_as<int32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int8> val) noexcept {
#  if DPL_SIMD_X86_SSE4_1
    return _mm_cvtepi8_epi32(+val);
#  else
    return xmm::element_cast<To>(xmm::element_cast<int16>(val));
#  endif
}

template <same_as<int32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint8> val) noexcept {
#  if DPL_SIMD_X86_SSE4_1
    return _mm_cvtepu8_epi32(+val);
#  else
    return xmm::element_cast<To>(xmm::element_cast<int16>(val));
#  endif
}

template <same_as<int32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<float> val) noexcept {
    return _mm_cvttps_epi32(+val);
}

template <same_as<int32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<double> val) noexcept {
    return _mm_cvttpd_epi32(+val);
}

template <same_as<int32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::bfloat16> val) noexcept {
    return xmm::element_cast<To>(xmm::element_cast<float>(val));
}

template <same_as<int32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvttph_epi32(+val);
#  else
    return xmm::element_cast<To>(xmm::element_cast<float>(val));
#  endif
}

#  if DPL_SIMD_X86_AVX512VL

#    if DPL_SIMD_X86_AVX512FP16

template <same_as<int32> To, imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int32> src, cmask_t<int32, M> mask,
        vector<ext::float16> val) noexcept {
    return _mm_mask_cvttph_epi32(+src, M, +val);
}

template <same_as<int32> To, imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int32, M> mask, vector<ext::float16> val) noexcept {
    return _mm_maskz_cvttph_epi32(M, +val);
}
#    endif

#    if DPL_SIMD_X86_AVX512F

template <same_as<int32> To, imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int32> src, cmask_t<int32, M> mask, vector<int8> val) noexcept {
    return _mm_mask_cvtepi8_epi32(+src, M, +val);
}

template <same_as<int32> To, imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int32, M> mask, vector<int8> val) noexcept {
    return _mm_maskz_cvtepi8_epi32(M, +val);
}

template <same_as<int32> To, imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int32> src, cmask_t<int32, M> mask, vector<uint8> val) noexcept {
    return _mm_mask_cvtepu8_epi32(+src, M, +val);
}

template <same_as<int32> To, imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int32, M> mask, vector<uint8> val) noexcept {
    return _mm_maskz_cvtepu8_epi32(M, +val);
}

template <same_as<int32> To, imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int32> src, cmask_t<int32, M> mask, vector<int16> val) noexcept {
    return _mm_mask_cvtepi16_epi32(+src, M, +val);
}

template <same_as<int32> To, imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int32, M> mask, vector<int16> val) noexcept {
    return _mm_maskz_cvtepi16_epi32(M, +val);
}

template <same_as<int32> To, imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int32> src, cmask_t<int32, M> mask,
        vector<uint16> val) noexcept {
    return _mm_mask_cvtepu16_epi32(+src, M, +val);
}

template <same_as<int32> To, imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int32, M> mask, vector<uint16> val) noexcept {
    return _mm_maskz_cvtepu16_epi32(M, +val);
}

template <same_as<int32> To, imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int32> src, cmask_t<int32, M> mask, vector<int64> val) noexcept {
    return _mm_mask_cvtepi64_epi32(+src, M, +val);
}

template <same_as<int32> To, imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int32, M> mask, vector<int64> val) noexcept {
    return _mm_maskz_cvtepi64_epi32(M, +val);
}

#    endif // if DPL_SIMD_X86_AVX512F

#  endif // if DPL_SIMD_X86_AVX512VL

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
