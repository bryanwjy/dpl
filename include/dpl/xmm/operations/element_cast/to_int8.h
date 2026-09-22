// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/zero.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <same_as<int8> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int16> val) noexcept {
#  if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi16_epi8(+val);
#  else
    auto const masked = _mm_and_si128(+val, +xmm::broadcast<int16>(0xff));
    return _mm_packus_epi16(masked, +xmm::broadcast<int>(dx::zero));
#  endif
}

template <same_as<int8> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int32> val) noexcept {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi32_epi8(+val);
#  else
    auto const masked = _mm_and_si128(+val, +xmm::broadcast<int>(0xff));
    auto const zero = +xmm::broadcast<int>(dx::zero);
    return _mm_packus_epi16(_mm_packus_epi32(masked, zero), zero);
#  endif
}

template <same_as<int8> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int64> val) noexcept {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi64_epi8(+val);
#  else
    auto const qwords = _mm_and_si128(+val, +xmm::broadcast<int64>(0xff));
    // Extract low 32 bits of each 64-bit lane
    auto const dwords = _mm_shuffle_epi32(qwords, _MM_SHUFFLE(3, 1, 2, 0));
    auto const zero = +xmm::broadcast<int>(dx::zero);
    return _mm_packus_epi16(_mm_packus_epi32(dwords, zero), zero);
#  endif
}

#  if DPL_SIMD_X86_AVX512VL

#    if DPL_SIMD_X86_AVX512F

template <same_as<int8> To, imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int8> src, cmask_t<int8, M> mask, vector<int32> val) noexcept {
    return _mm_mask_cvtepi32_epi8(+src, M, +val);
}

template <same_as<int8> To, imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int8, M> mask, vector<int32> val) noexcept {
    return _mm_maskz_cvtepi32_epi8(M, +val);
}

template <same_as<int8> To, imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int8> src, cmask_t<int8, M> mask, vector<int64> val) noexcept {
    return _mm_mask_cvtepi64_epi8(+src, M, +val);
}

template <same_as<int8> To, imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int8, M> mask, vector<int64> val) noexcept {
    return _mm_maskz_cvtepi64_epi8(M, +val);
}

#    endif // if DPL_SIMD_X86_AVX512F

#    if DPL_SIMD_X86_AVX512BW

template <same_as<int8> To, imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<int8> src, cmask_t<int8, M> mask, vector<int16> val) noexcept {
    return _mm_mask_cvtepi16_epi8(+src, M, +val);
}

template <same_as<int8> To, imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<int8, M> mask, vector<int16> val) noexcept {
    return _mm_maskz_cvtepi16_epi8(M, +val);
}

#    endif

#  endif // if DPL_SIMD_X86_AVX512VL

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
