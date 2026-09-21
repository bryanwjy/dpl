// Copyright 2025 Bryan Wong
#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE2

#  include "dpl/xmm/operations/element_cast/to_float.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <same_as<ext::bfloat16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<float> val) noexcept {
#  if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtneps_pbh(+val);
#  else
    // deal with nan & inf
    auto const fval = +val;
    auto const ival = _mm_castps_si128(fval);
    auto const zero = +xmm::broadcast<int>(dx::zero);
    auto const isnan =
        _mm_unpacklo_epi16(_mm_castps_si128(_mm_cmpunord_ps(fval, fval)), zero);
    // round to nearest even
    auto const lsb =
        _mm_and_si128(_mm_srli_epi32(ival, 16), +xmm::broadcast<int>(1));
    auto const low = +xmm::broadcast<int>(0x7fff);
    auto const bias = _mm_add_epi32(low, lsb);
    auto const result =
        _mm_or_si128(_mm_srli_epi32(_mm_add_epi32(ival, bias), 16), isnan);
    return __DPL bit_cast<__m128bh>(_mm_packus_epi32(result, zero));
#  endif
}

template <same_as<ext::bfloat16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<double> val) noexcept {
    return xmm::element_cast<To>(xmm::element_cast<float>(val));
}

template <same_as<ext::bfloat16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> val) noexcept {
    auto const hi =
        vector<ext::float16>(__DPL bit_cast<__m128h>(_mm_unpackhi_epi64(
            __DPL bit_cast<__m128i>(+val), +xmm::broadcast<int>(dx::zero))));
    auto const left = xmm::element_cast<To>(xmm::element_cast<float>(val));
    auto const right = xmm::element_cast<To>(xmm::element_cast<float>(hi));
    return __DPL bit_cast<__m128bh>(_mm_unpacklo_epi64(
        __DPL bit_cast<__m128i>(+left), __DPL bit_cast<__m128i>(+right)));
}

template <same_as<ext::bfloat16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int64> val) noexcept {
    return xmm::element_cast<To>(xmm::element_cast<float>(val));
}

template <same_as<ext::bfloat16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint64> val) noexcept {
    return xmm::element_cast<To>(xmm::element_cast<float>(val));
}

template <same_as<ext::bfloat16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int32> val) noexcept {
    return xmm::element_cast<To>(xmm::element_cast<float>(val));
}

template <same_as<ext::bfloat16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint32> val) noexcept {
    return xmm::element_cast<To>(xmm::element_cast<float>(val));
}

namespace details {
template <integral E>
requires common_size_with<int16, E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<ext::bfloat16> to_bfloat16(vector<E> val) noexcept {
    auto const hi =
        vector<E>(_mm_unpackhi_epi64(+val, +xmm::broadcast<int>(dx::zero)));
    auto const left =
        xmm::element_cast<ext::bfloat16>(xmm::element_cast<float>(val));
    auto const right =
        xmm::element_cast<ext::bfloat16>(xmm::element_cast<float>(hi));
    return __DPL bit_cast<__m128bh>(_mm_unpacklo_epi64(
        __DPL bit_cast<__m128i>(+left), __DPL bit_cast<__m128i>(+right)));
}
} // namespace details

template <same_as<ext::bfloat16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<int16> val) noexcept {
    return details::to_bfloat16(val);
}

template <same_as<ext::bfloat16> To>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<uint16> val) noexcept {
    return details::to_bfloat16(val);
}

#  if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(vector<ext::float16> src,
        cmask_t<ext::float16, M> mask, vector<float> val) noexcept {
    return _mm_mask_cbtneps_pbh(+src, M, +val);
}

template <same_as<ext::float16> To, imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<To>
    DPL_VECTORCALL element_cast(
        dx::zero_t, cmask_t<ext::float16, M> mask, vector<float> val) noexcept {
    return _mm_maskz_cbtneps_pbh(M, +val);
}
#  endif

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
