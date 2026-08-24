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
inline mask<int8> DPL_VECTORCALL cmpeq(
    vector<int8> lhs, vector<int8> rhs) noexcept {
    return _mm_cmpeq_epi8(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint8> DPL_VECTORCALL cmpeq(
    vector<uint8> lhs, vector<uint8> rhs) noexcept {
    return _mm_cmpeq_epi8(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int16> DPL_VECTORCALL cmpeq(
    vector<int16> lhs, vector<int16> rhs) noexcept {
    return _mm_cmpeq_epi16(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint16> DPL_VECTORCALL cmpeq(
    vector<uint16> lhs, vector<uint16> rhs) noexcept {
    return _mm_cmpeq_epi16(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int32> DPL_VECTORCALL cmpeq(
    vector<int32> lhs, vector<int32> rhs) noexcept {
    return _mm_cmpeq_epi32(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint32> DPL_VECTORCALL cmpeq(
    vector<uint32> lhs, vector<uint32> rhs) noexcept {
    return _mm_cmpeq_epi32(+lhs, +rhs);
}

#  if DPL_SIMD_X86_SSE4_1

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int64> DPL_VECTORCALL cmpeq(
    vector<int64> lhs, vector<int64> rhs) noexcept {
    return _mm_cmpeq_epi64(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint64> DPL_VECTORCALL cmpeq(
    vector<uint64> lhs, vector<uint64> rhs) noexcept {
    return _mm_cmpeq_epi64(+lhs, +rhs);
}

#  else

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<int64> DPL_VECTORCALL cmpeq(
    vector<int64> lhs, vector<int64> rhs) noexcept {
    auto const result = _mm_cmpeq_epi32(+lhs, +rhs);
    return _mm_and_si128(
        result, _mm_shuffle_epi32(result, _MM_SHUFFLE(2, 3, 0, 1)));
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<uint64> DPL_VECTORCALL cmpeq(
    vector<uint64> lhs, vector<uint64> rhs) noexcept {
    auto const result = _mm_cmpeq_epi32(+lhs, +rhs);
    return _mm_and_si128(
        result, _mm_shuffle_epi32(result, _MM_SHUFFLE(2, 3, 0, 1)));
}

#  endif

inline mask<float> DPL_VECTORCALL cmpeq(
    vector<float> lhs, vector<float> rhs) noexcept {
    return _mm_cmpeq_ps(+lhs, +rhs);
}

inline mask<double> DPL_VECTORCALL cmpeq(
    vector<double> lhs, vector<double> rhs) noexcept {
    return _mm_cmpeq_pd(+lhs, +rhs);
}

#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL

inline mask<ext::float16> DPL_VECTORCALL cmpeq(
    vector<ext::float16> lhs, vector<ext::float16> rhs) noexcept {
    return _mm_movm_epi16(_mm_cmp_ph_mask(+lhs, +rhs, _CMP_EQ_OQ));
}

template <imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<ext::float16>
    DPL_VECTORCALL cmpeq(cmask_t<ext::float16, M>, vector<ext::float16> lhs,
        vector<ext::float16> rhs) noexcept {
    return _mm_movm_epi16(_mm_mask_cmp_ph_mask(M, +lhs, +rhs, _CMP_EQ_OQ));
}

#  endif

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<ext::bfloat16> DPL_VECTORCALL cmpeq(
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
    auto const lo = _mm_castps_si128(_mm_cmpeq_ps(llhs, lrhs));
    auto const hi = _mm_castps_si128(_mm_cmpeq_ps(hlhs, hrhs));
    return __DPL bit_cast<__m128bh>(_mm_packs_epi32(lo, hi));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<E> cmpeq(mask<E> lhs, mask<E> rhs) noexcept {
    if constexpr (integral<E>) {
        return _mm_cmpeq_epi8(+lhs, +rhs);
    } else {
        using sint_t DPL_NODEBUG = signed_representation_t<E>;
        return xmm::cmpeq(
            xmm::reinterpret<sint_t>(lhs), xmm::reinterpret<sint_t>(rhs));
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> cmpeq(abi_tag, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::cmpeq(lhs, rhs); }
{
    return xmm::cmpeq(lhs, rhs);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> cmpeq(abi_tag, mask<E> lhs, mask<E> rhs) noexcept
requires requires { xmm::cmpeq(lhs, rhs); }
{
    return xmm::cmpeq(lhs, rhs);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> cmpeq(
    abi_tag, cmask_t<E, M> mask, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::cmpeq(mask, lhs, rhs); }
{
    return xmm::cmpeq(mask, lhs, rhs);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
