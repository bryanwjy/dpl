// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

#  include "dpl/xmm/operations/compare/cmplt.h"
#  include "dpl/xmm/operations/element_cast/fallback.h" // IWYU pragma: export
#  include "dpl/xmm/operations/element_cast/to_int16.h"
#  include "dpl/xmm/operations/reinterpret.h"
#  include "dpl/xmm/operations/select.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/msb.h"
#    include "dpl/core/immediate/constants/zero.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <same_as<uint16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::bfloat16> val) noexcept {
    auto const error = xmm::broadcast<int16>(dx::msb);
    auto const sresult = xmm::element_cast<int16>(val);
    auto const zero = xmm::broadcast<int16>(dx::zero);
    auto const rep = xmm::reinterpret<int16>(val);
    return +xmm::select(xmm::cmplt(rep, zero), error, sresult);
}

template <same_as<uint16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    auto const error = xmm::broadcast<int16>(dx::msb);
    auto const zero = xmm::broadcast<ext::float16>(dx::zero);
    auto const neg = _mm_cmp_ph(+val, +zero, _CMP_LT_OQ);
    return _mm_mask_blend_epi16(neg, _mm_cvttph_epi16(+val), +error);
#  else
    auto const error = xmm::broadcast<int16>(dx::msb);
    auto const sresult = xmm::element_cast<int16>(val);
    auto const zero = xmm::broadcast<int16>(dx::zero);
    auto const rep = xmm::reinterpret<int16>(val);
    return +xmm::select(xmm::cmplt(rep, zero), error, sresult);
#  endif
}

#  if DPL_SIMD_X86_AVX512VL

#    if DPL_SIMD_X86_AVX512FP16
template <same_as<uint16> To, imask_t<uint16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint16> src, cmask_t<uint16, M> mask,
        vector<ext::float16> val) noexcept {
    return _mm_mask_cvttph_epu16(+src, M, +val);
}

template <same_as<uint16> To, imask_t<uint16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(dx::zero_t, cmask_t<uint16, M> mask,
        vector<ext::float16> val) noexcept {
    return _mm_maskz_cvttph_epu16(M, +val);
}
#    endif // if DPL_SIMD_X86_AVX512FP16

#    if DPL_SIMD_X86_AVX512F
template <same_as<uint16> To, imask_t<uint16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint16> src, cmask_t<uint16, M> mask,
        vector<int32> val) noexcept {
    return _mm_mask_cvtepi32_epu16(+src, M, +val);
}

template <same_as<uint16> To, imask_t<uint16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<uint16, M> mask, vector<int32> val) noexcept {
    return _mm_maskz_cvtepi32_epu16(M, +val);
}

template <same_as<uint16> To, imask_t<uint16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint16> src, cmask_t<uint16, M> mask,
        vector<int64> val) noexcept {
    return _mm_mask_cvtepi64_epu16(+src, M, +val);
}

template <same_as<uint16> To, imask_t<uint16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<uint16, M> mask, vector<int64> val) noexcept {
    return _mm_maskz_cvtepi64_epu16(M, +val);
}

#    endif // if DPL_SIMD_X86_AVX512F

#  endif // if DPL_SIMD_X86_AVX512VL

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
