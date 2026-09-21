// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

#  include "dpl/xmm/operations/arithmetic/to_float.h"
#  include "dpl/xmm/operations/element_cast/fallback.h" // IWYU pragma: export

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/msb.h"
#    include "dpl/core/immediate/constants/zero.h"
#    include "dpl/core/numbers/floating_point_traits.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <same_as<uint64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<float> val) noexcept {
#  if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
    return _mm_cvttps_epu64(+val);
#  else
    return _mm_unpacklo_epi64(
        _mm_cvtsi64_si128(static_cast<int64>(val[imm<0>])),
        _mm_cvtsi64_si128(static_cast<int64>(val[imm<1>])));
#  endif
}

template <same_as<uint64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<double> val) noexcept {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvttpd_epu64(+val);
#  else
    return _mm_set_epi64x(
        static_cast<To>(val[imm<1>]), static_cast<To>(val[imm<0>]));
#  endif
}

template <same_as<uint64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::bfloat16> val) noexcept {
    return xmm::element_cast<To>(xmm::element_cast<float>(val));
}

template <same_as<uint64> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvttph_epu64(+val);
#  else
    return xmm::element_cast<To>(xmm::element_cast<float>(val));
#  endif
}

#  if DPL_SIMD_X86_AVX512VL

#    if DPL_SIMD_X86_AVX512FP16
template <same_as<uint64> To, imask_t<uint64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint64> src, cmask_t<uint64, M> mask,
        vector<ext::float16> val) noexcept {
    return _mm_mask_cvttph_epu64(+src, M, +val);
}

template <same_as<uint64> To, imask_t<uint64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(dx::zero_t, cmask_t<uint64, M> mask,
        vector<ext::float16> val) noexcept {
    return _mm_maskz_cvttph_epu64(M, +val);
}
#    endif // if DPL_SIMD_X86_AVX512FP16

#    if DPL_SIMD_X86_AVX512F
template <same_as<uint64> To, imask_t<uint64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint64> src, cmask_t<uint64, M> mask,
        vector<float> val) noexcept {
    return _mm_mask_cvttps_epu64(+src, M, +val);
}

template <same_as<uint64> To, imask_t<uint64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<uint64, M> mask, vector<float> val) noexcept {
    return _mm_maskz_cvttps_epu64(M, +val);
}

template <same_as<uint64> To, imask_t<uint64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint64> src, cmask_t<uint64, M> mask,
        vector<double> val) noexcept {
    return _mm_mask_cvttpd_epu64(+src, M, +val);
}

template <same_as<uint64> To, imask_t<uint64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<uint64, M> mask, vector<double> val) noexcept {
    return _mm_maskz_cvttpd_epu64(M, +val);
}
#    endif // if DPL_SIMD_X86_AVX512F
#  endif   // if DPL_SIMD_X86_AVX512VL

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
