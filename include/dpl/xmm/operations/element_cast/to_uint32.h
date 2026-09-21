// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

#  include "dpl/xmm/operations/arithmetic/to_float.h"
#  include "dpl/xmm/operations/bitwise/bwandnot.h"
#  include "dpl/xmm/operations/compare/cmplt.h"
#  include "dpl/xmm/operations/element_cast/fallback.h" // IWYU pragma: export
#  include "dpl/xmm/operations/reinterpret.h"
#  include "dpl/xmm/operations/select.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/msb.h"
#    include "dpl/core/immediate/constants/zero.h"
#    include "dpl/core/numbers/floating_point_traits.h"
#    include "dpl/std/utility/to_signed.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <same_as<uint32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<float> val) noexcept {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvttps_epu32(+val);
#  else
    // Based on generated code from clang
    constexpr int32 magic = 0x4f000000;
    auto xmm0 = +val;
    auto xmm2 = _mm_castsi128_ps(+xmm::broadcast<int>(magic));
    auto xmm1 = _mm_cmple_ps(xmm2, xmm0);
    xmm2 = _mm_and_ps(xmm1, xmm2);
    xmm1 = _mm_castps_si128(_mm_slli_epi32(_mm_castps_si128(xmm1), 31));
    xmm0 = _mm_sub_ps(xmm0, xmm2);
    xmm0 = _mm_cvttps_epi32(xmm0);
    return _mm_xor_ps(xmm0, xmm1);
#  endif
}

template <same_as<uint32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<double> val) noexcept {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvttpd_epu32(+val);
#  else
    return _mm_set_epi32(0, 0, static_cast<uint32>(val[imm<1>]),
        static_cast<uint32>(val[imm<0>]));
#  endif
}

template <same_as<uint32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::bfloat16> val) noexcept {
    return xmm::element_cast<To>(xmm::element_cast<float>(val));
}

template <same_as<uint32> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvttph_epu32(+val);
#  else
    constexpr auto mantissa_width = floating_point_traits<float>::digits - 1;
    constexpr auto hidden_bit = 1u << mantissa_width;
    constexpr auto bias = floating_point_traits<float>::exponent_bias;
    constexpr auto exp = __DPL to_signed(
        __DPL to_underlying(floating_point_traits<float>::exponent_mask));
    // There's likely a faster way, but this is just easier :p
    auto const f32 = xmm::element_cast<float>(val);
    auto const s32 = xmm::reinterpret<int32>(f32);
    auto const valid = xmm::bwandnot( //
        xmm::cmplt(s32, xmm::broadcast<int32>(exp)),
        xmm::cmplt(s32, xmm::broadcast<int32>(dx::zero)));
    auto const result = xmm::select(
        valid, xmm::element_cast<int32>(f32), xmm::broadcast<int32>(dx::msb));

    return xmm::reinterpret<uint32>(result);
#  endif
}

#  if DPL_SIMD_X86_AVX512VL

#    if DPL_SIMD_X86_AVX512FP16
template <same_as<uint32> To, imask_t<uint32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint32> src, cmask_t<uint32, M> mask,
        vector<ext::float16> val) noexcept {
    return _mm_mask_cvttph_epu32(+src, M, +val);
}

template <same_as<uint32> To, imask_t<uint32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(dx::zero_t, cmask_t<uint32, M> mask,
        vector<ext::float16> val) noexcept {
    return _mm_maskz_cvttph_epu32(M, +val);
}
#    endif // if DPL_SIMD_X86_AVX512FP16

#    if DPL_SIMD_X86_AVX512F
template <same_as<uint32> To, imask_t<uint32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint32> src, cmask_t<uint32, M> mask,
        vector<float> val) noexcept {
    return _mm_mask_cvttps_epu32(+src, M, +val);
}

template <same_as<uint32> To, imask_t<uint32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<uint32, M> mask, vector<float> val) noexcept {
    return _mm_maskz_cvttps_epu32(M, +val);
}

template <same_as<uint32> To, imask_t<uint32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint32> src, cmask_t<uint32, M> mask,
        vector<double> val) noexcept {
    return _mm_mask_cvttpd_epu32(+src, M, +val);
}

template <same_as<uint32> To, imask_t<uint32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<uint32, M> mask, vector<double> val) noexcept {
    return _mm_maskz_cvttpd_epu32(M, +val);
}
#    endif // if DPL_SIMD_X86_AVX512F
#  endif   // if DPL_SIMD_X86_AVX512VL

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
