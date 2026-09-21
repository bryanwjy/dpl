// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

#  include "dpl/xmm/operations/arithmetic/to_float.h"
#  include "dpl/xmm/operations/element_cast/fallback.h" // IWYU pragma: export
#  include "dpl/xmm/operations/element_cast/to_int32.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/zero.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <same_as<int64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int8> val) noexcept {
#  if DPL_SIMD_X86_SSE4_1
    return _mm_cvtepi8_epi64(+val);
#  else
    return xmm::element_cast<To>(xmm::element_cast<int32>(val));
#  endif
}

template <same_as<int64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint8> val) noexcept {
#  if DPL_SIMD_X86_SSE4_1
    return _mm_cvtepu8_epi64(+val);
#  else
    return xmm::element_cast<To>(xmm::element_cast<int32>(val));
#  endif
}

template <same_as<int64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int16> val) noexcept {
#  if DPL_SIMD_X86_SSE4_1
    return _mm_cvtepi16_epi64(+val);
#  else
    return xmm::element_cast<To>(xmm::element_cast<int32>(val));
#  endif
}

template <same_as<int64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint16> val) noexcept {
#  if DPL_SIMD_X86_SSE4_1
    return _mm_cvtepu16_epi64(+val);
#  else
    return xmm::element_cast<To>(xmm::element_cast<int32>(val));
#  endif
}

template <same_as<int64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int32> val) noexcept {
#  if DPL_SIMD_X86_SSE4_1
    return _mm_cvtepi32_epi64(+val);
#  else
    auto const neg = _mm_cmplt_epi32(+val, +xmm::broadcast<int32>(dx::zero));
    return _mm_unpacklo_epi32(+val, neg);
#  endif
}

template <same_as<int64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint32> val) noexcept {
#  if DPL_SIMD_X86_SSE4_1
    return _mm_cvtepu32_epi64(+val);
#  else
    return _mm_unpacklo_epi32(+val, +xmm::broadcast<int32>(dx::zero));
#  endif
}

template <same_as<int64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<float> val) noexcept {
#  if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
    return _mm_cvttps_epi64(+val);
#  else
    return _mm_unpacklo_epi64(
        _mm_cvtsi64_si128(static_cast<int64>(val[imm<0zu>])),
        _mm_cvtsi64_si128(static_cast<int64>(val[imm<1zu>])));
#  endif
}

template <same_as<int64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<double> val) noexcept {
#  if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
    return _mm_cvttpd_epi64(+val);
#  else
    return _mm_set_epi64x(
        static_cast<To>(val[imm<1>]), static_cast<To>(val[imm<0>]));
#  endif
}

template <same_as<int64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::bfloat16> val) noexcept {
    return xmm::element_cast<To>(xmm::element_cast<float>(val));
}

template <same_as<int64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvttph_epi64(+val);
#  else
    return xmm::element_cast<To>(xmm::element_cast<float>(val));
#  endif
}

#  if DPL_SIMD_X86_AVX512VL

#    if DPL_SIMD_X86_AVX512FP16

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int64> src, cmask_t<int64, M> mask,
        vector<ext::float16> val) noexcept {
    return _mm_mask_cvttph_epi64(+src, M, +val);
}

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int64, M> mask, vector<ext::float16> val) noexcept {
    return _mm_maskz_cvttph_epi64(M, +val);
}
#    endif

#    if DPL_SIMD_X86_AVX512F

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int64> src, cmask_t<int64, M> mask, vector<int8> val) noexcept {
    return _mm_mask_cvtepi8_epi64(+src, M, +val);
}

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int64, M> mask, vector<int8> val) noexcept {
    return _mm_maskz_cvtepi8_epi64(M, +val);
}

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int64> src, cmask_t<int64, M> mask, vector<uint8> val) noexcept {
    return _mm_mask_cvtepu8_epi64(+src, M, +val);
}

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int64, M> mask, vector<uint8> val) noexcept {
    return _mm_maskz_cvtepu8_epi64(M, +val);
}

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int64> src, cmask_t<int64, M> mask, vector<int16> val) noexcept {
    return _mm_mask_cvtepi16_epi64(+src, M, +val);
}

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int64, M> mask, vector<int16> val) noexcept {
    return _mm_maskz_cvtepi16_epi64(M, +val);
}

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int64> src, cmask_t<int64, M> mask,
        vector<uint16> val) noexcept {
    return _mm_mask_cvtepu16_epi64(+src, M, +val);
}

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int64, M> mask, vector<uint16> val) noexcept {
    return _mm_maskz_cvtepu16_epi64(M, +val);
}

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int64> src, cmask_t<int64, M> mask, vector<int32> val) noexcept {
    return _mm_mask_cvtepi32_epi64(+src, M, +val);
}

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int64, M> mask, vector<int32> val) noexcept {
    return _mm_maskz_cvtepi32_epi64(M, +val);
}

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int64> src, cmask_t<int64, M> mask,
        vector<uint32> val) noexcept {
    return _mm_mask_cvtepu32_epi64(+src, M, +val);
}

template <same_as<int64> To, imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int64, M> mask, vector<uint32> val) noexcept {
    return _mm_maskz_cvtepu32_epi64(M, +val);
}

#    endif // if DPL_SIMD_X86_AVX512F

#  endif // if DPL_SIMD_X86_AVX512VL

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
