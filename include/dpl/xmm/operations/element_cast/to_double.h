// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

#  include "dpl/xmm/operations/arithmetic/multiply.h"
#  include "dpl/xmm/operations/bitwise.h"
#  include "dpl/xmm/operations/compare/cmplt.h"
#  include "dpl/xmm/operations/element_cast/to_float.h"
#  include "dpl/xmm/operations/element_cast/to_int64.h"
#  include "dpl/xmm/operations/reinterpret.h"
#  include "dpl/xmm/operations/select.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/msb.h"
#    include "dpl/core/immediate/constants/zero.h"
#    include "dpl/core/numbers/floating_point_traits.h"
#    include "dpl/std/utility/bitset.h"
#    include "dpl/std/utility/to_signed.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<float> val) noexcept {
    return _mm_cvtps_pd(+val);
}

template <same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtph_pd(+val);
#  else
    auto const arg = xmm::element_cast<int64>(xmm::reinterpret<uint16>(val));
    auto const signmask = xmm::broadcast<int64>(0x8000);
    auto const signs = xmm::reinterpret<double>(
        xmm::bwshift_left<48>(xmm::bwand(arg, signmask)));
    auto const parg = xmm::bwshift_left<42>(xmm::bwandnot(arg, signmask));

    constexpr auto exp = __DPL to_signed(__DPL to_underlying(
        bitset<64>(floating_point_traits<ext::float16>::exponent_mask) << 42));

    auto const vexp = xmm::broadcast<int64>(exp);
    auto const isfinite = xmm::cmplt(parg, vexp);

    constexpr auto inf64 = __DPL to_signed(
        __DPL to_underlying(floating_point_traits<double>::exponent_mask));
    auto const vinf64 = xmm::broadcast<int64>(inf64);
    auto const nonfinite =
        xmm::select(isfinite, dx::zero, xmm::bwor(vinf64, parg));

    auto const shift = xmm::broadcast<double>(0x1p1008);
    auto const shifted = xmm::multiply(xmm::reinterpret<double>(parg), shift);
    auto const abs_f64 =
        xmm::bwor(xmm::reinterpret<double>(nonfinite), shifted);
    return xmm::bwor(abs_f64, signs);
#  endif
}

template <same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::bfloat16> val) noexcept {
    return xmm::element_cast<To>(xmm::element_cast<float>(val));
}

template <same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int8> val) noexcept {
    auto const lo = _mm_cvtepi16_epi32(_mm_srai_epi16(
        _mm_unpacklo_epi8(+xmm::broadcast<int>(dx::zero), +val), 8));
    return _mm_cvtepi32_pd(lo);
}

template <same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint8> val) noexcept {
    auto const lo = _mm_cvtepu16_epi32(
        _mm_unpacklo_epi8(+val, +xmm::broadcast<int>(dx::zero)));
    return _mm_cvtepi32_pd(lo);
}

template <same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int16> val) noexcept {
    return _mm_cvtepi32_pd(_mm_cvtepi16_epi32(+val));
}

template <same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint16> val) noexcept {
    return _mm_cvtepi32_pd(_mm_cvtepu16_epi32(+val));
}

template <same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int32> val) noexcept {
    return _mm_cvtepi32_pd(+val);
}

template <same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint32> val) noexcept {
    auto const large = _mm_cmplt_epi32(+val, +xmm::broadcast<int>(dx::zero));
    auto const islolarge = _mm_castsi128_pd(_mm_unpacklo_epi32(large, large));
    return _mm_add_pd(
        _mm_and_pd(islolarge, _mm_set1_pd(0x1p32)), _mm_cvtepi32_pd(+val));
}

template <same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int64> val) noexcept {
#  if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi64_pd(+val);
#  else
    // Without AVX512, it is simply not worth it to vectorize
    return _mm_setr_pd(static_cast<double>(_mm_extract_epi64(+val, 0)),
        static_cast<double>(_mm_extract_epi64(+val, 1)));
#  endif
}

template <same_as<double> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint64> val) noexcept {
#  if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepu64_pd(+val);
#  else
    // Without AVX512, it is simply not worth it to vectorize
    return _mm_setr_pd(
        static_cast<double>( __DPL to_unsigned(_mm_extract_epi64(+val, 0))),
        static_cast<double>( __DPL to_unsigned(_mm_extract_epi64(+val, 1))));
#  endif
}

#  if DPL_SIMD_X86_AVX512VL

#    if DPL_SIMD_X86_AVX512FP16
template <same_as<double> To, imask_t<To> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<double> src, cmask_t<To, M> mask,
        vector<ext::float16> val) noexcept {
    return _mm_mask_cvtph_pd(+src, M, +val);
}

template <same_as<double> To, imask_t<To> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<To, M> mask, vector<ext::float16> val) noexcept {
    return _mm_maskz_cvtph_pd(M, +val);
}
#    endif

#    if DPL_SIMD_X86_AVX512F
template <same_as<double> To, imask_t<To> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<double> src, cmask_t<To, M> mask, vector<int32> val) noexcept {
    return _mm_mask_cvtepi32_ps(+src, M, +val);
}

template <same_as<double> To, imask_t<To> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<To, M> mask, vector<int32> val) noexcept {
    return _mm_maskz_cvtepi32_ps(M, +val);
}

template <same_as<double> To, imask_t<To> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<double> src, cmask_t<To, M> mask, vector<uint32> val) noexcept {
    return _mm_mask_cvtepu32_ps(+src, M, +val);
}

template <same_as<double> To, imask_t<To> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<To, M> mask, vector<uint32> val) noexcept {
    return _mm_maskz_cvtepu32_ps(M, +val);
}
#    endif

#    if DPL_SIMD_X86_AVX512DQ
template <same_as<double> To, imask_t<To> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<double> src, cmask_t<To, M> mask, vector<int64> val) noexcept {
    return _mm_mask_cvtepi64_ps(+src, M, +val);
}

template <same_as<double> To, imask_t<To> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<To, M> mask, vector<int64> val) noexcept {
    return _mm_maskz_cvtepi64_ps(M, +val);
}

template <same_as<double> To, imask_t<To> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<double> src, cmask_t<To, M> mask, vector<uint64> val) noexcept {
    return _mm_mask_cvtepu64_ps(+src, M, +val);
}

template <same_as<double> To, imask_t<To> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<To, M> mask, vector<uint64> val) noexcept {
    return _mm_maskz_cvtepu64_ps(M, +val);
}
#    endif
#  endif

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
