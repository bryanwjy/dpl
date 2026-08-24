// Copyright 2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2
#  include "dpl/xmm/operations/element_cast.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/msb.h"
#    include "dpl/core/immediate/constants/zero.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int8> DPL_VECTORCALL cmplt(
    vector<int8> lhs, vector<int8> rhs) noexcept {
    return _mm_cmplt_epi8(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint8> DPL_VECTORCALL cmplt(
    vector<uint8> lhs, vector<uint8> rhs) noexcept {
    auto const bias = +xmm::broadcast<int8>(dx::msb);
    return _mm_cmplt_epi8(_mm_xor_si128(+lhs, bias), _mm_xor_si128(+rhs, bias));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int16> DPL_VECTORCALL cmplt(
    vector<int16> lhs, vector<int16> rhs) noexcept {
    return _mm_cmplt_epi16(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint16> DPL_VECTORCALL cmplt(
    vector<uint16> lhs, vector<uint16> rhs) noexcept {
    auto const bias = +xmm::broadcast<int16>(dx::msb);
    return _mm_cmplt_epi16(
        _mm_xor_si128(+lhs, bias), _mm_xor_si128(+rhs, bias));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int32> DPL_VECTORCALL cmplt(
    vector<int32> lhs, vector<int32> rhs) noexcept {
    return _mm_cmplt_epi32(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint32> DPL_VECTORCALL cmplt(
    vector<uint32> lhs, vector<uint32> rhs) noexcept {
    auto const bias = +xmm::broadcast<int32>(dx::msb);
    return _mm_cmplt_epi32(
        _mm_xor_si128(+lhs, bias), _mm_xor_si128(+rhs, bias));
}

#  if DPL_SIMD_X86_SSE4_2

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int64> DPL_VECTORCALL cmplt(
    vector<int64> lhs, vector<int64> rhs) noexcept {
    return _mm_cmpgt_epi64(+rhs, +lhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint64> DPL_VECTORCALL cmplt(
    vector<uint64> lhs, vector<uint64> rhs) noexcept {
#    if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_movm_epi64(_mm_cmp_epu64_mask(+lhs, +rhs, _MM_CMPINT_LT));
#    else
    auto const bias = +xmm::broadcast<int64>(dx::msb);
    return _mm_cmpgt_epi64(
        _mm_xor_si128(+rhs, bias), _mm_xor_si128(+lhs, bias));
#    endif
}

#  else // if DPL_SIMD_X86_SSE4_2

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int64> DPL_VECTORCALL cmplt(
    vector<int64> lhs, vector<int64> rhs) noexcept {
    auto r =
        _mm_and_si128(_mm_cmpeq_epi32(+lhs, +rhs), _mm_sub_epi64(+lhs, +rhs));
    r = _mm_or_si128(r, _mm_cmplt_epi32(+lhs, +rhs));
    return _mm_shuffle_epi32(_mm_srai_epi32(r, 31), _MM_SHUFFLE(3, 3, 1, 1));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint64> DPL_VECTORCALL cmplt(
    vector<uint64> lhs, vector<uint64> rhs) noexcept {
    auto const bias = +xmm::broadcast<int64>(dx::msb);
    return xmm::cmplt(vector<int64>(_mm_xor_si128(+lhs, bias)),
        vector<int64>(_mm_xor_si128(+rhs, bias)));
}

#  endif // if DPL_SIMD_X86_SSE4_2

inline mask<float> DPL_VECTORCALL cmplt(
    vector<float> lhs, vector<float> rhs) noexcept {
    return _mm_cmplt_ps(+lhs, +rhs);
}

inline mask<double> DPL_VECTORCALL cmplt(
    vector<double> lhs, vector<double> rhs) noexcept {
    return _mm_cmplt_pd(+lhs, +rhs);
}

#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL

inline mask<ext::float16> DPL_VECTORCALL cmplt(
    vector<ext::float16> lhs, vector<ext::float16> rhs) noexcept {
    return _mm_movm_epi16(_mm_cmp_ph_mask(+lhs, +rhs, _CMP_LT_OQ));
}

template <imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<ext::float16>
    DPL_VECTORCALL cmplt(cmask_t<ext::float16, M>, vector<ext::float16> lhs,
        vector<ext::float16> rhs) noexcept {
    return _mm_movm_epi16(_mm_mask_cmp_ph_mask(M, +lhs, +rhs, _CMP_LT_OQ));
}

#  endif // if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<ext::bfloat16> DPL_VECTORCALL cmplt(
    vector<ext::bfloat16> lhs, vector<ext::bfloat16> rhs) noexcept {
    auto const zero = +xmm::broadcast<int>(dx::zero);
    auto const llhs = _mm_castsi128_ps(
        _mm_unpacklo_epi16(zero, __DPL bit_cast<__m128i>(+lhs)));
    auto const lrhs = _mm_castsi128_ps(
        _mm_unpacklo_epi16(zero, __DPL bit_cast<__m128i>(+rhs)));
    auto const hlhs = _mm_castsi128_ps(
        _mm_unpackhi_epi16(zero, __DPL bit_cast<__m128i>(+lhs)));
    auto const hrhs = _mm_castsi128_ps(
        _mm_unpackhi_epi16(zero, __DPL bit_cast<__m128i>(+rhs)));
    auto const lo = _mm_castps_si128(_mm_cmplt_ps(llhs, lrhs));
    auto const hi = _mm_castps_si128(_mm_cmplt_ps(hlhs, hrhs));
    return __DPL bit_cast<__m128bh>(_mm_packs_epi32(lo, hi));
}

#  if DPL_SIMD_X86_AVX512VL

#    if DPL_SIMD_X86_AVX512F
// Only unsigned integral variants are worth it, all others comparisons only end
// up adding more instructions
template <imask_t<uint64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint64>
    DPL_VECTORCALL cmplt(
        cmask_t<uint64, M>, vector<uint64> lhs, vector<uint64> rhs) noexcept {
    return _mm_movm_epi64(
        _mm_mask_cmp_epu64_mask(M, +lhs, +rhs, _MM_CMPINT_LT));
}

template <imask_t<uint32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint32>
    DPL_VECTORCALL cmplt(
        cmask_t<uint32, M>, vector<uint32> lhs, vector<uint32> rhs) noexcept {
    return _mm_movm_epi32(
        _mm_mask_cmp_epu32_mask(M, +lhs, +rhs, _MM_CMPINT_LT));
}
#    endif // if DPL_SIMD_X86_AVX512F

#    if DPL_SIMD_X86_AVX512BW
template <imask_t<uint16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint16>
    DPL_VECTORCALL cmplt(
        cmask_t<uint16, M>, vector<uint16> lhs, vector<uint16> rhs) noexcept {
    return _mm_movm_epi16(
        _mm_mask_cmp_epu16_mask(M, +lhs, +rhs, _MM_CMPINT_LT));
}

template <imask_t<uint8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint8>
    DPL_VECTORCALL cmplt(
        cmask_t<uint8, M>, vector<uint8> lhs, vector<uint8> rhs) noexcept {
    return _mm_movm_epi8(_mm_mask_cmp_epu8_mask(M, +lhs, +rhs, _MM_CMPINT_LT));
}
#    endif // if DPL_SIMD_X86_AVX512BW
#  endif   // if DPL_SIMD_X86_AVX512VL

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> cmplt(abi_tag, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::cmplt(lhs, rhs); }
{
    return xmm::cmplt(lhs, rhs);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> cmplt(
    abi_tag, cmask_t<E, M> mask, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::cmplt(mask, lhs, rhs); }
{
    return xmm::cmplt(mask, lhs, rhs);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
