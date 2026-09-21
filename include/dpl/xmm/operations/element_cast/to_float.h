// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

#  include "dpl/xmm/operations/arithmetic/multiply.h"
#  include "dpl/xmm/operations/arithmetic/to_float.h"
#  include "dpl/xmm/operations/bitwise.h"
#  include "dpl/xmm/operations/compare/cmplt.h"
#  include "dpl/xmm/operations/element_cast/to_int32.h"
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

template <same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<double> val) noexcept {
    return _mm_cvtpd_ps(+val);
}

template <same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> val) noexcept {
#  if DPL_SIMD_X86_F16C
    return _mm_cvtph_ps(+val);
#  else
    auto const arg = xmm::element_cast<int32>(xmm::reinterpret<uint16>(val));

    auto const signmask = xmm::broadcast<int32>(dx::msb_v<uint16>);
    auto const signs = xmm::reinterpret<float>(
        xmm::bwshift_left<16>(xmm::bwand(arg, signmask)));

    auto const parg = xmm::bwshift_left<13>(xmm::bwandnot(arg, signmask));

    constexpr auto exp = __DPL to_signed( __DPL to_underlying(
        bitset<32>(floating_point_traits<ext::float16>::exponent_mask) << 13));
    auto const vexp = xmm::broadcast<int32>(exp);
    auto const isfinite = xmm::cmplt(parg, vexp);

    constexpr auto inf32 = __DPL to_signed(
        __DPL to_underlying(floating_point_traits<float>::exponent_mask));
    auto const vinf32 = xmm::broadcast<int32>(inf32);
    auto const nonfinite =
        xmm::select(isfinite, dx::zero, xmm::bwor(vinf32, parg));

    auto const shift = xmm::broadcast<float>(0x1p112f);
    auto const shifted = xmm::multiply(xmm::reinterpret<float>(parg), shift);
    auto const abs_f32 = xmm::bwor(xmm::reinterpret<float>(nonfinite), shifted);
    return xmm::bwor(abs_f32, signs);
#  endif
}

template <same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::bfloat16> val) noexcept {
#  if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtpbh_ps(+val);
#  else
    auto const isrc = xmm::element_cast<int32>(xmm::reinterpret<uint16>(val));
    return xmm::reinterpret<float>(xmm::bwshift_left<16>(isrc));
#  endif
}

template <same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int8> val) noexcept {
    auto const lo = _mm_cvtepi16_epi32(_mm_srai_epi16(
        _mm_unpacklo_epi8(+xmm::broadcast<int>(dx::zero), +val), 8));
    return _mm_cvtepi32_ps(lo);
}

template <same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint8> val) noexcept {
    auto const lo = _mm_cvtepu16_epi32(
        _mm_unpacklo_epi8(+val, +xmm::broadcast<int>(dx::zero)));
    return _mm_cvtepi32_ps(lo);
}

template <same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int16> val) noexcept {
    return _mm_cvtepi32_ps(_mm_cvtepi16_epi32(+val));
}

template <same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint16> val) noexcept {
    return _mm_cvtepi32_ps(_mm_cvtepu16_epi32(+val));
}

template <same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int32> val) noexcept {
    return _mm_cvtepi32_ps(+val);
}

template <same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint32> val) noexcept {
    constexpr auto max = integral_traits<int32>::max_value;
    auto const gt = _mm_cmpgt_epi32(+val, +xmm::broadcast<int32>(max));
    return _mm_add_ps(
        _mm_cvtepi32_ps(+val), _mm_and_ps(gt, +xmm::broadcast<float>(0x1p32f)));
}

template <same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int64> val) noexcept {
#  if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi64_ps(+val);
#  else
    auto const zero = +xmm::broadcast<int>(dx::zero);
    auto const hisign = _mm_and_si128(+xmm::broadcast<int64>(1ll << 63), +val);
    auto const losign =
        _mm_castsi128_ps(_mm_srli_epi64(_mm_castps_si128(hisign), 32));

    val = xmm::abs(val);
    auto const ishizero = _mm_cmpeq_epi32(+val, zero);
    auto const hicorrection = _mm_andnot_si128(
        ishizero, _mm_or_si128(+xmm::broadcast<int64>(32ll << 55), hisign));
    auto const islarge = _mm_castsi128_ps(_mm_cmplt_epi32(+val, zero));
    auto const locorrection =
        _mm_and_ps(islarge, +xmm::broadcast<float>(0x1p32f));
    auto hilohilo =
        _mm_add_ps(_mm_castsi128_ps(_mm_add_epi64(
                       _mm_castps_si128(_mm_cvtepi32_ps(+val)), hicorrection)),
            locorrection);
    hilohilo = _mm_xor_ps(hilohilo, losign);
    auto const hihilolo = _mm_castps_si128(
        _mm_shuffle_ps(hilohilo, hilohilo, _MM_SHUFFLE(3, 1, 2, 0)));
    return _mm_add_ps(_mm_castsi128_ps(_mm_unpacklo_epi64(hihilolo, zero)),
        _mm_castsi128_ps(_mm_unpackhi_epi64(hihilolo, zero)));
#  endif
}

template <same_as<float> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint64> val) noexcept {
#  if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepu64_ps(+val);
#  else
    auto const zero = +xmm::broadcast<int>(dx::zero);
    auto const islarge = _mm_castsi128_ps(_mm_cmplt_epi32(+val, zero));
    auto const correction =
        _mm_and_ps(islarge, +xmm::broadcast<float>(0x1p32f));
    auto const ishizero = _mm_cmpeq_epi32(+val, zero);
    auto const hilohilo = _mm_add_ps(
        _mm_castsi128_ps(_mm_add_epi64(_mm_castps_si128(_mm_cvtepi32_ps(+val)),
            _mm_andnot_si128(ishizero, +xmm::broadcast<int64>(32ll << 55)))),
        correction);
    auto const hihilolo = _mm_castps_si128(
        _mm_shuffle_ps(hilohilo, hilohilo, _MM_SHUFFLE(3, 1, 2, 0)));

    return _mm_add_ps(_mm_castsi128_ps(_mm_unpacklo_epi64(hihilolo, zero)),
        _mm_castsi128_ps(_mm_unpackhi_epi64(hihilolo, zero)));
#  endif
}

#  if DPL_SIMD_X86_AVX512VL

#    if DPL_SIMD_X86_AVX512F
template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<float> src, cmask_t<float, M> mask,
        vector<double> val) noexcept {
    return _mm_mask_cvtpd_ps(+src, M, +val);
}

template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<float, M> mask, vector<double> val) noexcept {
    return _mm_maskz_cvtpd_ps(M, +val);
}

template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<float> src, cmask_t<float, M> mask, vector<int32> val) noexcept {
    return _mm_mask_cvtepi32_ps(+src, M, +val);
}

template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<float, M> mask, vector<int32> val) noexcept {
    return _mm_maskz_cvtepi32_ps(M, +val);
}

template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<float> src, cmask_t<float, M> mask,
        vector<uint32> val) noexcept {
    constexpr auto max = integral_traits<int32>::max_value;
    auto const gt = _mm_cmpgt_epi32(+val, +xmm::broadcast<int32>(max));
    return _mm_mask_add_ps(+src, M, _mm_maskz_cvtepi32_ps(M, +val),
        _mm_and_ps(gt, +xmm::broadcast<float>(0x1p32f)));
}

template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<float, M> mask, vector<uint32> val) noexcept {
    constexpr auto max = integral_traits<int32>::max_value;
    auto const gt = _mm_cmpgt_epi32(+val, +xmm::broadcast<int32>(max));
    return _mm_maskz_add_ps(M, _mm_maskz_cvtepi32_ps(M, +val),
        _mm_and_ps(gt, +xmm::broadcast<float>(0x1p32f)));
}
#    endif

#    if DPL_SIMD_X86_AVX512FP16
template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<float> src, cmask_t<float, M> mask,
        vector<ext::float16> val) noexcept {
    return _mm_mask_cvtph_ps(+src, M, +val);
}

template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<float, M> mask, vector<ext::float16> val) noexcept {
    return _mm_maskz_cvtph_ps(M, +val);
}
#    endif

#    if DPL_SIMD_X86_AVX512BF16
template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<float> src, cmask_t<float, M> mask,
        vector<ext::bfloat16> val) noexcept {
    return _mm_mask_cvtpbh_ps(+src, M, +val);
}

template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(dx::zero_t, cmask_t<float, M> mask,
        vector<ext::bfloat16> val) noexcept {
    return _mm_maskz_cvtpbh_ps(M, +val);
}
#    endif

#    if DPL_SIMD_X86_AVX512DQ
template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        vector<float> src, cmask_t<float, M> mask, vector<int64> val) noexcept {
    return _mm_mask_cvtepi64_ps(+src, M, +val);
}

template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<float, M> mask, vector<int64> val) noexcept {
    return _mm_maskz_cvtepi64_ps(M, +val);
}

template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<float> src, cmask_t<float, M> mask,
        vector<uint64> val) noexcept {
    return _mm_mask_cvtepu64_ps(+src, M, +val);
}

template <same_as<float> To, imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<float, M> mask, vector<uint64> val) noexcept {
    return _mm_maskz_cvtepu64_ps(M, +val);
}
#    endif
#  endif

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
