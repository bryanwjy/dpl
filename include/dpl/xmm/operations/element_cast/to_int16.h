// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

#  include "dpl/xmm/operations/element_cast/fallback.h" // IWYU pragma: export

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/zero.h"
#    include "dpl/std/bit/bit_cast.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <same_as<int32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> val) noexcept;
template <same_as<int32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::bfloat16> val) noexcept;

template <same_as<int16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int8> val) noexcept {
#  if DPL_SIMD_X86_SSE4_1
    return _mm_cvtepi8_epi16(+val);
#  else
    auto const neg = _mm_cmplt_epi8(+val, +xmm::broadcast<int8>(dx::zero));
    return _mm_unpacklo_epi8(+val, neg);
#  endif
}

template <same_as<int16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint8> val) noexcept {
#  if DPL_SIMD_X86_SSE4_1
    return _mm_cvtepu8_epi16(+val);
#  else
    return _mm_unpacklo_epi8(+val, +xmm::broadcast<int8>(dx::zero));
#  endif
}

template <same_as<int16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int32> val) noexcept {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi32_epi16(+val);
#  else
    auto const dwords = _mm_and_si128(+val, +xmm::broadcast<int>(0xffff));
    return _mm_packus_epi32(dwords, +xmm::broadcast<int>(dx::zero));
#  endif
}

template <same_as<int16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int64> val) noexcept {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi64_epi16(+val);
#  else
    auto const qwords = _mm_and_si128(+val, +xmm::broadcast<int64>(0xffff));
    // Extract low 32 bits of each 64-bit lane
    auto const dwords = _mm_shuffle_epi32(qwords, _MM_SHUFFLE(3, 1, 2, 0));
    return _mm_packus_epi32(dwords, +xmm::broadcast<int>(dx::zero));
#  endif
}

template <same_as<int16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::bfloat16> val) noexcept {
    auto const hi =
        vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(_mm_unpackhi_epi64(
            __DPL bit_cast<__m128i>(+val), +xmm::broadcast<int>(dx::zero))));
    auto const lo_mask = +xmm::broadcast<int>(0xffff);
    auto const left = xmm::element_cast<To>(xmm::element_cast<int32>(val));
    auto const right = xmm::element_cast<To>(xmm::element_cast<int32>(hi));
    return _mm_unpacklo_epi64(+left, +right);
}

template <same_as<int16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvttph_epi16(+val);
#  else
    auto const hi =
        vector<ext::float16>(__DPL bit_cast<__m128h>(_mm_unpackhi_epi64(
            __DPL bit_cast<__m128i>(+val), +xmm::broadcast<int>(dx::zero))));
    auto const lo_mask = +xmm::broadcast<int>(0xffff);
    auto const left = xmm::element_cast<To>(xmm::element_cast<int32>(val));
    auto const right = xmm::element_cast<To>(xmm::element_cast<int32>(hi));
    return _mm_unpacklo_epi64(+left, +right);
#  endif
}

#  if DPL_SIMD_X86_AVX512VL

#    if DPL_SIMD_X86_AVX512FP16
template <same_as<int16> To, imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int16> src, cmask_t<int16, M> mask,
        vector<ext::float16> val) noexcept {
    return _mm_mask_cvttph_epi16(+src, M, +val);
}

template <same_as<int16> To, imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int16, M> mask, vector<ext::float16> val) noexcept {
    return _mm_maskz_cvttph_epi16(M, +val);
}
#    endif // if DPL_SIMD_X86_AVX512FP16

#    if DPL_SIMD_X86_AVX512F
template <same_as<int16> To, imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int16> src, cmask_t<int16, M> mask, vector<int32> val) noexcept {
    return _mm_mask_cvtepi32_epi16(+src, M, +val);
}

template <same_as<int16> To, imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int16, M> mask, vector<int32> val) noexcept {
    return _mm_maskz_cvtepi32_epi16(M, +val);
}

template <same_as<int16> To, imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int16> src, cmask_t<int16, M> mask, vector<int64> val) noexcept {
    return _mm_mask_cvtepi64_epi16(+src, M, +val);
}

template <same_as<int16> To, imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int16, M> mask, vector<int64> val) noexcept {
    return _mm_maskz_cvtepi64_epi16(M, +val);
}

#    endif // if DPL_SIMD_X86_AVX512F

#    if DPL_SIMD_X86_AVX512BW

template <same_as<int16> To, imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int16> src, cmask_t<int16, M> mask, vector<int8> val) noexcept {
    return _mm_mask_cvtepi8_epi16(+src, M, +val);
}

template <same_as<int16> To, imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int16, M> mask, vector<int8> val) noexcept {
    return _mm_maskz_cvtepi8_epi16(M, +val);
}

template <same_as<int16> To, imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int16> src, cmask_t<int16, M> mask, vector<uint8> val) noexcept {
    return _mm_mask_cvtepu8_epi16(+src, M, +val);
}

template <same_as<int16> To, imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int16, M> mask, vector<uint8> val) noexcept {
    return _mm_maskz_cvtepu8_epi16(M, +val);
}

#    endif

#  endif // if DPL_SIMD_X86_AVX512VL

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
