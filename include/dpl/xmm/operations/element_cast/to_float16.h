// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

#  include "dpl/xmm/operations/arithmetic/abs.h"
#  include "dpl/xmm/operations/bitwise.h"
#  include "dpl/xmm/operations/compare/cmplt.h"
#  include "dpl/xmm/operations/element_cast/to_float.h"
#  include "dpl/xmm/operations/element_cast/to_int16.h"
#  include "dpl/xmm/operations/element_cast/to_int32.h"
#  include "dpl/xmm/operations/element_cast/to_uint16.h"
#  include "dpl/xmm/operations/reinterpret.h"
#  include "dpl/xmm/operations/select.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/msb.h"
#    include "dpl/core/immediate/constants/zero.h"
#    include "dpl/core/numbers/floating_point_traits.h"
#    include "dpl/std/bit/bit_cast.h"
#    include "dpl/std/utility/bitset.h"
#    include "dpl/std/utility/to_signed.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
namespace details {
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::float16> DPL_VECTORCALL to_postive_inthalf(
    vector<float> f32) noexcept {
    auto const bits = xmm::reinterpret<uint32>(f32);
    auto const mantissa =
        xmm::bwand(xmm::bwshift_right<13>(bits), xmm::broadcast<uint32>(0x3ff));
    auto const exp = _mm_subs_epu16(
        +xmm::bwshift_right<23>(bits), +xmm::broadcast<int>(112));
    auto const bexp = vector<uint32>(_mm_slli_epi32(exp, 10));
    return xmm::reinterpret<ext::float16>(xmm::select(
        mask<float>(_mm_cmpge_ps(+f32, +xmm::broadcast<float>(0x1p16f))),
        xmm::broadcast<uint32>(0x7c00), xmm::bwor(bexp, mantissa)));
}
} // namespace details

template <same_as<ext::float16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint64> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepu64_ph(+val);
#  else
    auto const zero = +xmm::broadcast<int>(dx::zero);
    auto const all = +xmm::broadcast<int64>(0xffff);
    auto const isinf = mask<uint16>(_mm_packs_epi32(
        _mm_packs_epi32(_mm_cmpgt_epi64(+val, all), zero), zero));
    auto const trunc32 =
        vector<int32>(_mm_packus_epi32(_mm_and_si128(+val, all), zero));
    auto const trunc =
        details::to_postive_inthalf(xmm::element_cast<float>(trunc32));
    return xmm::select(
        isinf, xmm::broadcast<ext::float16>(dx::infinity), trunc);
#  endif
}

template <same_as<ext::float16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int64> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi64_ph(+val);
#  else
    auto const zero = +xmm::broadcast<int>(dx::zero);
    auto const abs = xmm::reinterpret<uint64>(xmm::abs(val));
    auto const vsign16 = vector<uint16>(
        _mm_and_si128(_mm_packs_epi32(_mm_packs_epi32(+val, zero), zero),
            +xmm::broadcast<int16>(dx::msb)));

    return xmm::reinterpret<ext::float16>(xmm::bit_fill(
        vsign16, xmm::element_cast<To>(xmm::reinterpret<uint64>(abs))));
#  endif
}

template <same_as<ext::float16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint32> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepu32_ph(+val);
#  else
    auto const zero = +xmm::broadcast<int>(dx::zero);
    auto const all = +xmm::broadcast<int>(0xffff);
    auto const isinf =
        mask<uint16>(_mm_packs_epi32(_mm_cmpgt_epi32(+val, all), zero));
    auto const clamped = vector<uint32>(_mm_and_si128(+val, all));
    auto const trunc =
        details::to_postive_inthalf(xmm::element_cast<float>(clamped));
    return xmm::select(
        isinf, xmm::broadcast<ext::float16>(dx::infinity), trunc);
#  endif
}

template <same_as<ext::float16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int32> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi32_ph(+val);
#  else
    auto const zero = +xmm::broadcast<int>(dx::zero);
    auto const abs = xmm::reinterpret<uint32>(xmm::abs(val));
    auto const vsign16 = vector<uint16>(_mm_and_si128(
        _mm_packs_epi32(+val, zero), +xmm::broadcast<int16>(dx::msb)));
    return xmm::reinterpret<ext::float16>(xmm::bit_fill(
        vsign16, xmm::element_cast<To>(xmm::reinterpret<uint32>(abs))));
#  endif
}

template <same_as<ext::float16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint16> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepu16_ph(+val);
#  else
    auto const zero = +xmm::broadcast<int>(dx::zero);
    auto const lo = xmm::element_cast<float>(
        vector<uint32>(_mm_unpacklo_epi16(+val, zero)));
    auto const hi = xmm::element_cast<float>(
        vector<uint32>(_mm_unpackhi_epi16(+val, zero)));
    auto const lo16 = xmm::reinterpret<float>(details::to_postive_inthalf(lo));
    auto const hi16 = xmm::reinterpret<float>(details::to_postive_inthalf(hi));
    return xmm::reinterpret<ext::float16>(
        vector<float>(_mm_shuffle_ps(lo16, hi16, _MM_SHUFFLE(1, 0, 1, 0))));
#  endif
}

template <same_as<ext::float16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int16> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi16_ph(+val);
#  else
    auto const sign = xmm::reinterpret<ext::float16>(
        vector<uint32>(_mm_and_si128(+val, +xmm::broadcast<int16>(0x8000))));
    auto const abs =
        xmm::element_cast<To>(xmm::reinterpret<uint16>(xmm::abs(val)));
    return xmm::bwor(sign, abs);
#  endif
}

template <same_as<ext::float16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint8> val) noexcept {
    return xmm::element_cast<To>(xmm::element_cast<uint16>(val));
}

template <same_as<ext::float16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int8> val) noexcept {
    return xmm::element_cast<To>(xmm::element_cast<int16>(val));
}

template <same_as<ext::float16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<float> val) noexcept {
#  if DPL_SIMD_X86_F16C
    return __DPL bit_cast<__m128h>(
        _mm_cvtps_ph(+val, _MM_FROUND_TO_NEAREST_INT));
#  else
    auto const i32 = xmm::reinterpret<int32>(val);
    auto const msb32 = xmm::broadcast<int32>(dx::msb);
    auto const sign16 = xmm::bwshift_right<16>(
        xmm::reinterpret<uint32>(xmm::bwand(msb32, i32)));
    auto const abs = xmm::reinterpret<float>(xmm::bwandnot(i32, msb32));
    auto const shifted = xmm::reinterpret<uint32>(
        xmm::multiply(abs, xmm::broadcast<float>(0x1p-112f)));

    auto const rounded = xmm::bwshift_right<13>(
        xmm::add(xmm::bwand(xmm::bwshift_right<13>(shifted),
                     xmm::broadcast<uint32>(dx::one)),
            xmm::add(shifted, xmm::broadcast<uint32>(0xfffu))));

    auto const limit = xmm::broadcast<float>(0x1p16f);
    auto const isinf = mask<float>(_mm_cmpge_ps(+abs, +limit));
    auto const isnan = xmm::bwandnot(vector<float>(_mm_cmpunord_ps(+val, +val)),
        xmm::reinterpret<float>(msb32));

    constexpr auto infval = static_cast<int32>(
        __DPL bit_cast<int16>(dx::infinity_v<ext::float16>));
    auto const inf16 =
        xmm::reinterpret<ext::float16>(xmm::broadcast<int32>(infval));
    auto const f16 = xmm::select(isinf, xmm::reinterpret<float>(inf16),
        xmm::reinterpret<float>(rounded));
    auto const result =
        xmm::bwor(sign16, xmm::reinterpret<uint32>(xmm::bwor(isnan, f16)));
    return xmm::reinterpret<ext::float16>(vector<uint32>(
        _mm_packus_epi32(+result, +xmm::broadcast<int>(dx::zero))));
#  endif
}

template <same_as<ext::float16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<double> val) noexcept {
#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtpd_ph(+val);
#  else
    auto const i64 = xmm::reinterpret<int64>(val);
    auto const msb64 = xmm::broadcast<int64>(dx::msb);
    auto const sign16 = xmm::bwshift_right<48>(
        xmm::reinterpret<uint64>(xmm::bwand(msb64, i64)));
    auto const abs = xmm::reinterpret<double>(xmm::bwandnot(i64, msb64));
    auto const shifted = xmm::reinterpret<uint64>(
        xmm::multiply(abs, xmm::broadcast<double>(0x1p-1008)));
    constexpr auto round_mask = (0x1ull << 41) - 1ull;
    auto const rounded = xmm::bwshift_right<42>(
        xmm::add(xmm::bwand(xmm::bwshift_right<42>(shifted),
                     xmm::broadcast<uint64>(dx::one)),
            xmm::add(shifted, xmm::broadcast<uint64>(round_mask))));
    auto const limit = xmm::broadcast<double>(0x1p16);
    auto const isinf = mask<double>(_mm_cmpge_pd(+abs, +limit));
    auto const isnan = xmm::bwand(vector<double>(_mm_cmpunord_pd(+val, +val)),
        xmm::reinterpret<double>(xmm::broadcast<int64>(0x7fff)));

    constexpr auto infval = static_cast<int64>(
        __DPL bit_cast<int16>(dx::infinity_v<ext::float16>));
    auto const inf16 =
        xmm::reinterpret<ext::float16>(xmm::broadcast<int64>(infval));
    auto const f16 = xmm::select(isinf, xmm::reinterpret<double>(inf16),
        xmm::reinterpret<double>(rounded));
    auto const result =
        xmm::bwor(sign16, xmm::reinterpret<uint64>(xmm::bwor(isnan, f16)));
    auto const zero = +xmm::broadcast<int>(dx::zero);
    // The cast here is just to ditribute the bits into place
    return xmm::reinterpret<ext::float16>(vector<ext::float16>(
        _mm_packus_epi32(_mm_shuffle_ps(_mm_castsi128_ps(+result),
                             _mm_castsi128_ps(zero), _MM_SHUFFLE(2, 0, 2, 0)),
            zero)));
#  endif
}

template <same_as<ext::float16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::bfloat16> val) noexcept {
    auto const hi =
        vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(_mm_unpackhi_epi64(
            __DPL bit_cast<__m128i>(+val), +xmm::broadcast<int>(dx::zero))));
    auto const left = xmm::element_cast<To>(xmm::element_cast<float>(val));
    auto const right = xmm::element_cast<To>(xmm::element_cast<float>(hi));
    return __DPL bit_cast<__m128h>(_mm_unpacklo_epi64(
        __DPL bit_cast<__m128i>(+left), __DPL bit_cast<__m128i>(+right)));
}

#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> src,
        cmask_t<ext::float16, M> mask, vector<int64> val) noexcept {
    return _mm_mask_cvtepi64_ph(+src, M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<ext::float16, M> mask, vector<int64> val) noexcept {
    return _mm_maskz_cvtepi64_ph(M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> src,
        cmask_t<ext::float16, M> mask, vector<uint64> val) noexcept {
    return _mm_mask_cvtepu64_ph(+src, M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(dx::zero_t, cmask_t<ext::float16, M> mask,
        vector<uint32> val) noexcept {
    return _mm_maskz_cvtepu32_ph(M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> src,
        cmask_t<ext::float16, M> mask, vector<int32> val) noexcept {
    return _mm_mask_cvtepi32_ph(+src, M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<ext::float16, M> mask, vector<int32> val) noexcept {
    return _mm_maskz_cvtepi32_ph(M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> src,
        cmask_t<ext::float16, M> mask, vector<uint32> val) noexcept {
    return _mm_mask_cvtepu32_ph(+src, M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(dx::zero_t, cmask_t<ext::float16, M> mask,
        vector<uint32> val) noexcept {
    return _mm_maskz_cvtepu32_ph(M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> src,
        cmask_t<ext::float16, M> mask, vector<int16> val) noexcept {
    return _mm_mask_cvtepi16_ph(+src, M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<ext::float16, M> mask, vector<int16> val) noexcept {
    return _mm_maskz_cvtepi16_ph(M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> src,
        cmask_t<ext::float16, M> mask, vector<uint16> val) noexcept {
    return _mm_mask_cvtepu16_ph(+src, M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(dx::zero_t, cmask_t<ext::float16, M> mask,
        vector<uint16> val) noexcept {
    return _mm_maskz_cvtepu16_ph(M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> src,
        cmask_t<ext::float16, M> mask, vector<double> val) noexcept {
    return _mm_mask_cvtpd_ph(+src, M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(dx::zero_t, cmask_t<ext::float16, M> mask,
        vector<double> val) noexcept {
    return _mm_maskz_cvtpd_ph(M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> src,
        cmask_t<ext::float16, M> mask, vector<float> val) noexcept {
    return _mm_mask_cvtxps_ph(+src, M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<ext::float16, M> mask, vector<float> val) noexcept {
    return _mm_maskz_cvtxps_ph(M, +val);
}
#  endif

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
