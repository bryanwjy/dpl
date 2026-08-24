// Copyright 2025 Bryan Wong

#pragma once
#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  if !DPL_MODULES
#    include "dpl/xmm/basic/abi.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int8> DPL_VECTORCALL min(
    vector<int8> lhs, vector<int8> rhs) noexcept {
    return _mm_min_epi8(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint8> DPL_VECTORCALL min(
    vector<uint8> lhs, vector<uint8> rhs) noexcept {
    return _mm_min_epu8(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int16> DPL_VECTORCALL min(
    vector<int16> lhs, vector<int16> rhs) noexcept {
    return _mm_min_epi16(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint16> DPL_VECTORCALL min(
    vector<uint16> lhs, vector<uint16> rhs) noexcept {
    return _mm_min_epu16(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32> DPL_VECTORCALL min(
    vector<int32> lhs, vector<int32> rhs) noexcept {
    return _mm_min_epi32(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint32> DPL_VECTORCALL min(
    vector<uint32> lhs, vector<uint32> rhs) noexcept {
    return _mm_min_epu32(+lhs, +rhs);
}

#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int64> DPL_VECTORCALL min(
    vector<int64> lhs, vector<int64> rhs) noexcept {
    return _mm_min_epi64(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint64> DPL_VECTORCALL min(
    vector<uint64> lhs, vector<uint64> rhs) noexcept {
    return _mm_min_epu64(+lhs, +rhs);
}

#  endif

inline vector<float> DPL_VECTORCALL min(
    vector<float> lhs, vector<float> rhs) noexcept {
    return _mm_min_ps(+lhs, +rhs);
}

inline vector<double> DPL_VECTORCALL min(
    vector<double> lhs, vector<double> rhs) noexcept {
    return _mm_min_pd(+lhs, +rhs);
}

#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL

inline vector<ext::float16> DPL_VECTORCALL min(
    vector<ext::float16> lhs, vector<ext::float16> rhs) noexcept {
    return _mm_min_ph(+lhs, +rhs);
}

#  endif

#  if DPL_SIMD_X86_AVX512VL
#    if DPL_SIMD_X86_AVX512F

template <imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32>
    DPL_VECTORCALL min(vector<int32> src, cmask_t<E, M>, vector<int32> lhs,
        vector<int32> rhs) noexcept {
    return _mm_mask_min_epi32(+src, M, +lhs, +rhs);
}

template <imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32>
    DPL_VECTORCALL min(dx::zero_t, cmask_t<E, M>, vector<int32> lhs,
        vector<int32> rhs) noexcept {
    return _mm_maskz_min_epi32(M, +lhs, +rhs);
}

template <imask_t<uint32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint32>
    DPL_VECTORCALL min(vector<uint32> src, cmask_t<E, M>, vector<uint32> lhs,
        vector<uint32> rhs) noexcept {
    return _mm_mask_min_epu32(+src, M, +lhs, +rhs);
}

template <imask_t<uint32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint32>
    DPL_VECTORCALL min(dx::zero_t, cmask_t<E, M>, vector<uint32> lhs,
        vector<uint32> rhs) noexcept {
    return _mm_maskz_min_epu32(M, +lhs, +rhs);
}

template <imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int64>
    DPL_VECTORCALL min(vector<int64> src, cmask_t<E, M>, vector<int64> lhs,
        vector<int64> rhs) noexcept {
    return _mm_mask_min_epi64(+src, M, +lhs, +rhs);
}

template <imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int64>
    DPL_VECTORCALL min(dx::zero_t, cmask_t<E, M>, vector<int64> lhs,
        vector<int64> rhs) noexcept {
    return _mm_maskz_min_epi64(M, +lhs, +rhs);
}

template <imask_t<uint64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint64>
    DPL_VECTORCALL min(vector<uint64> src, cmask_t<E, M>, vector<uint64> lhs,
        vector<uint64> rhs) noexcept {
    return _mm_mask_min_epu64(+src, M, +lhs, +rhs);
}

template <imask_t<uint64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint64>
    DPL_VECTORCALL min(dx::zero_t, cmask_t<E, M>, vector<uint64> lhs,
        vector<uint64> rhs) noexcept {
    return _mm_maskz_min_epu64(M, +lhs, +rhs);
}

template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL min(vector<float> src, cmask_t<E, M>, vector<float> lhs,
        vector<float> rhs) noexcept {
    return _mm_mask_min_ps(+src, M, +lhs, +rhs);
}

template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL min(dx::zero_t, cmask_t<E, M>, vector<float> lhs,
        vector<float> rhs) noexcept {
    return _mm_maskz_min_ps(M, +lhs, +rhs);
}

template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double>
    DPL_VECTORCALL min(vector<double> src, cmask_t<E, M>, vector<double> lhs,
        vector<double> rhs) noexcept {
    return _mm_mask_min_pd(+src, M, +lhs, +rhs);
}

template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double>
    DPL_VECTORCALL min(dx::zero_t, cmask_t<E, M>, vector<double> lhs,
        vector<double> rhs) noexcept {
    return _mm_maskz_min_pd(M, +lhs, +rhs);
}

#    endif //    if DPL_SIMD_X86_AVX512F

#    if DPL_SIMD_X86_AVX512BW

template <imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int8>
    DPL_VECTORCALL min(vector<int8> src, cmask_t<E, M>, vector<int8> lhs,
        vector<int8> rhs) noexcept {
    return _mm_mask_min_epi8(+src, M, +lhs, +rhs);
}

template <imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int8>
    DPL_VECTORCALL min(dx::zero_t, cmask_t<E, M>, vector<int8> lhs,
        vector<int8> rhs) noexcept {
    return _mm_maskz_min_epi8(M, +lhs, +rhs);
}

template <imask_t<uint8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint8>
    DPL_VECTORCALL min(vector<uint8> src, cmask_t<E, M>, vector<uint8> lhs,
        vector<uint8> rhs) noexcept {
    return _mm_mask_min_epu8(+src, M, +lhs, +rhs);
}

template <imask_t<uint8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint8>
    DPL_VECTORCALL min(dx::zero_t, cmask_t<E, M>, vector<uint8> lhs,
        vector<uint8> rhs) noexcept {
    return _mm_maskz_min_epu8(M, +lhs, +rhs);
}

template <imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int16>
    DPL_VECTORCALL min(vector<int16> src, cmask_t<E, M>, vector<int16> lhs,
        vector<int16> rhs) noexcept {
    return _mm_mask_min_epi16(+src, M, +lhs, +rhs);
}

template <imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int16>
    DPL_VECTORCALL min(dx::zero_t, cmask_t<E, M>, vector<int16> lhs,
        vector<int16> rhs) noexcept {
    return _mm_maskz_min_epi16(M, +lhs, +rhs);
}

template <imask_t<uint16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint16>
    DPL_VECTORCALL min(vector<uint16> src, cmask_t<E, M>, vector<uint16> lhs,
        vector<uint16> rhs) noexcept {
    return _mm_mask_min_epu16(+src, M, +lhs, +rhs);
}

template <imask_t<uint16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint16>
    DPL_VECTORCALL min(dx::zero_t, cmask_t<E, M>, vector<uint16> lhs,
        vector<uint16> rhs) noexcept {
    return _mm_maskz_min_epu16(M, +lhs, +rhs);
}

#    endif // if DPL_SIMD_X86_AVX512BW

#    if DPL_SIMD_X86_AVX512FP16

template <imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::float16>
    DPL_VECTORCALL min(vector<ext::float16> src, cmask_t<E, M>,
        vector<ext::float16> lhs, vector<ext::float16> rhs) noexcept {
    return _mm_mask_min_ph(+src, M, +lhs, +rhs);
}

template <imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::float16>
    DPL_VECTORCALL min(dx::zero_t, cmask_t<E, M>, vector<ext::float16> lhs,
        vector<ext::float16> rhs) noexcept {
    return _mm_maskz_min_ph(M, +lhs, +rhs);
}

#    endif // if DPL_SIMD_X86_AVX512FP16

#  endif // if DPL_SIMD_X86_AVX512VL

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> min(abi_tag, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::min(lhs, rhs); }
{
    return xmm::min(lhs, rhs);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> min(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::min(src, mask, lhs, rhs); }
{
    return xmm::min(src, mask, lhs, rhs);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> min(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
    vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::min(zero, mask, lhs, rhs); }
{
    return xmm::min(zero, mask, lhs, rhs);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
