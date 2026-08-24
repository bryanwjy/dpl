// Copyright 2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2
#  include "dpl/xmm/operations/compare/cmpeq.h"
#  include "dpl/xmm/operations/element_cast.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/all_bits.h"
#    include "dpl/core/immediate/constants/msb.h"
#    include "dpl/core/immediate/constants/zero.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <integral E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<E>
    DPL_VECTORCALL cmpneq(vector<E> lhs, vector<E> rhs) noexcept {
    auto const all = +xmm::broadcast<E>(dx::all_bits);
    return _mm_andnot_si128(+xmm::cmpeq(lhs, rhs), all);
}

inline mask<float> DPL_VECTORCALL cmpneq(
    vector<float> lhs, vector<float> rhs) noexcept {
    return _mm_cmpneq_ps(+lhs, +rhs);
}

inline mask<double> DPL_VECTORCALL cmpneq(
    vector<double> lhs, vector<double> rhs) noexcept {
    return _mm_cmpneq_pd(+lhs, +rhs);
}

#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL

inline mask<ext::float16> DPL_VECTORCALL cmpneq(
    vector<ext::float16> lhs, vector<ext::float16> rhs) noexcept {
    return _mm_movm_epi16(_mm_cmp_ph_mask(+lhs, +rhs, _CMP_NEQ_OQ));
}

template <imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<ext::float16>
    DPL_VECTORCALL cmpneq(cmask_t<ext::float16, M>, vector<ext::float16> lhs,
        vector<ext::float16> rhs) noexcept {
    return _mm_movm_epi16(_mm_mask_cmp_ph_mask(M, +lhs, +rhs, _CMP_NEQ_OQ));
}

#  endif

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<ext::bfloat16> DPL_VECTORCALL cmpneq(
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
    auto const lo = _mm_castps_si128(_mm_cmpneq_ps(llhs, lrhs));
    auto const hi = _mm_castps_si128(_mm_cmpneq_ps(hlhs, hrhs));
    return __DPL bit_cast<__m128bh>(_mm_packs_epi32(lo, hi));
}

#  if DPL_SIMD_X86_AVX512VL
#    if DPL_SIMD_X86_AVX512F
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int64> DPL_VECTORCALL cmpneq(
    vector<int64> lhs, vector<int64> rhs) noexcept {
    return _mm_movm_epi64(_mm_cmp_epi64_mask(+lhs, +rhs, _MM_CMPINT_NE));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int32> DPL_VECTORCALL cmpneq(
    vector<int32> lhs, vector<int32> rhs) noexcept {
    return _mm_movm_epi32(_mm_cmp_epi32_mask(+lhs, +rhs, _MM_CMPINT_NE));
}

inline mask<uint64> DPL_VECTORCALL cmpneq(
    vector<uint64> lhs, vector<uint64> rhs) noexcept {
    return _mm_movm_epi64(_mm_cmp_epu64_mask(+lhs, +rhs, _MM_CMPINT_NE));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint32> DPL_VECTORCALL cmpneq(
    vector<uint32> lhs, vector<uint32> rhs) noexcept {
    return _mm_movm_epi32(_mm_cmp_epu32_mask(+lhs, +rhs, _MM_CMPINT_NE));
}

template <imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int64>
    DPL_VECTORCALL cmpneq(
        cmask_t<int64, M>, vector<int64> lhs, vector<int64> rhs) noexcept {
    return _mm_movm_epi64(
        _mm_mask_cmp_epi64_mask(M, +lhs, +rhs, _MM_CMPINT_NE));
}

template <imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int32>
    DPL_VECTORCALL cmpneq(
        cmask_t<int32, M>, vector<int32> lhs, vector<int32> rhs) noexcept {
    return _mm_movm_epi32(
        _mm_mask_cmp_epi32_mask(M, +lhs, +rhs, _MM_CMPINT_NE));
}

template <imask_t<uint64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint64>
    DPL_VECTORCALL cmpneq(
        cmask_t<uint64, M>, vector<uint64> lhs, vector<uint64> rhs) noexcept {
    return _mm_movm_epi64(
        _mm_mask_cmp_epu64_mask(M, +lhs, +rhs, _MM_CMPINT_NE));
}

template <imask_t<uint32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint32>
    DPL_VECTORCALL cmpneq(
        cmask_t<uint32, M>, vector<uint32> lhs, vector<uint32> rhs) noexcept {
    return _mm_movm_epi32(
        _mm_mask_cmp_epu32_mask(M, +lhs, +rhs, _MM_CMPINT_NE));
}
#    endif // if DPL_SIMD_X86_AVX512F

#    if DPL_SIMD_X86_AVX512BW
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int16> DPL_VECTORCALL cmpneq(
    vector<int16> lhs, vector<int16> rhs) noexcept {
    return _mm_movm_epi16(_mm_cmp_epi16_mask(+lhs, +rhs, _MM_CMPINT_NE));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int8> DPL_VECTORCALL cmpneq(
    vector<int8> lhs, vector<int8> rhs) noexcept {
    return _mm_movm_epi8(_mm_cmp_epi8_mask(+lhs, +rhs, _MM_CMPINT_NE));
}

inline mask<uint16> DPL_VECTORCALL cmpneq(
    vector<uint16> lhs, vector<uint16> rhs) noexcept {
    return _mm_movm_epi16(_mm_cmp_epu16_mask(+lhs, +rhs, _MM_CMPINT_NE));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint8> DPL_VECTORCALL cmpneq(
    vector<uint8> lhs, vector<uint8> rhs) noexcept {
    return _mm_movm_epi8(_mm_cmp_epu8_mask(+lhs, +rhs, _MM_CMPINT_NE));
}

template <imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int16>
    DPL_VECTORCALL cmpneq(
        cmask_t<int16, M>, vector<int16> lhs, vector<int16> rhs) noexcept {
    return _mm_movm_epi16(
        _mm_mask_cmp_epi16_mask(M, +lhs, +rhs, _MM_CMPINT_NE));
}

template <imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int8>
    DPL_VECTORCALL cmpneq(
        cmask_t<int8, M>, vector<int8> lhs, vector<int8> rhs) noexcept {
    return _mm_movm_epi8(_mm_mask_cmp_epi8_mask(M, +lhs, +rhs, _MM_CMPINT_NE));
}

template <imask_t<uint16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint16>
    DPL_VECTORCALL cmpneq(
        cmask_t<uint16, M>, vector<uint16> lhs, vector<uint16> rhs) noexcept {
    return _mm_movm_epi16(
        _mm_mask_cmp_epu16_mask(M, +lhs, +rhs, _MM_CMPINT_NE));
}

template <imask_t<uint8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint8>
    DPL_VECTORCALL cmpneq(
        cmask_t<uint8, M>, vector<uint8> lhs, vector<uint8> rhs) noexcept {
    return _mm_movm_epi8(_mm_mask_cmp_epu8_mask(M, +lhs, +rhs, _MM_CMPINT_NE));
}
#    endif // if DPL_SIMD_X86_AVX512BW
#  endif   // if DPL_SIMD_X86_AVX512VL

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<E> cmpneq(mask<E> lhs, mask<E> rhs) noexcept {
    if constexpr (integral<E>) {
        return _mm_xor_si128(+lhs, +rhs);
    } else {
        using sint_t DPL_NODEBUG = signed_representation_t<E>;
        return xmm::cmpneq(
            xmm::reinterpret<sint_t>(lhs), xmm::reinterpret<sint_t>(rhs));
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> cmpneq(abi_tag, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::cmpneq(lhs, rhs); }
{
    return xmm::cmpneq(lhs, rhs);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> cmpneq(abi_tag, mask<E> lhs, mask<E> rhs) noexcept
requires requires { xmm::cmpneq(lhs, rhs); }
{
    return xmm::cmpneq(lhs, rhs);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> cmpneq(
    abi_tag, cmask_t<E, M> mask, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::cmpneq(mask, lhs, rhs); }
{
    return xmm::cmpneq(mask, lhs, rhs);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
