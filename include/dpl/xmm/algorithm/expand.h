// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_AVX512VL

#  if !DPL_MODULES
#    include "dpl/std/type_traits/type_identity.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"
#    include "dpl/xmm/operations/reinterpret.h"

#    include <immintrin.h>
#  endif

#  if DPL_SIMD_X86_AVX512F

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double> DPL_VECTORCALL expand(mask_identity_t<double> mask,
    vector<double> arg, vector<double> src) noexcept {
    auto const vzero = xmm::broadcast<int64>(dx::zero);
    auto const kmask = _mm_cmp_epi64_mask(
        +xmm::reinterpret<int64>(mask), vzero, _MM_CMPINT_LT);
    return _mm_mask_expand_pd(+src, kmask, +arg);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float> DPL_VECTORCALL expand(mask_identity_t<float> mask,
    vector<float> arg, vector<float> src) noexcept {
    auto const vzero = xmm::broadcast<int32>(dx::zero);
    auto const kmask = _mm_cmp_epi32_mask(
        +xmm::reinterpret<int32>(mask), vzero, _MM_CMPINT_LT);
    return _mm_mask_expand_ps(+src, kmask, +arg);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int64> DPL_VECTORCALL expand(mask_identity_t<int64> mask,
    vector<int64> arg, vector<int64> src) noexcept {
    auto const vzero = xmm::broadcast<int64>(dx::zero);
    auto const kmask = _mm_cmp_epi64_mask(+mask, vzero, _MM_CMPINT_LT);
    return _mm_mask_expand_epi64(+src, kmask, +arg);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32> DPL_VECTORCALL expand(mask_identity_t<int32> mask,
    vector<int32> arg, vector<int32> src) noexcept {
    auto const vzero = xmm::broadcast<int32>(dx::zero);
    auto const kmask = _mm_cmp_epi32_mask(+mask, vzero, _MM_CMPINT_LT);
    return _mm_mask_expand_epi32(+src, kmask, +arg);
}

template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double>
    DPL_VECTORCALL expand(
        cmask_t<double, M>, vector<double> arg, vector<double> src) noexcept {
    return _mm_mask_expand_pd(+src, M, +arg);
}

template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL expand(
        cmask_t<float, M>, vector<float> arg, vector<float> src) noexcept {
    return _mm_mask_expand_ps(+src, M, +arg);
}

template <imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int64>
    DPL_VECTORCALL expand(
        cmask_t<int64, M>, vector<int64> arg, vector<int64> src) noexcept {
    return _mm_mask_expand_epi64(+src, M, +arg);
}

template <imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32>
    DPL_VECTORCALL expand(
        cmask_t<int32, M>, vector<int32> arg, vector<int32> src) noexcept {
    return _mm_mask_expand_epi32(+src, M, +arg);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double> DPL_VECTORCALL expand(mask_identity_t<double> mask,
    vector<double> arg, dx::zero_t zero) noexcept {
    auto const vzero = xmm::broadcast<int64>(zero);
    auto const kmask = _mm_cmp_epi64_mask(
        +xmm::reinterpret<int64>(mask), vzero, _MM_CMPINT_LT);
    return _mm_maskz_expand_pd(kmask, +arg);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float> DPL_VECTORCALL expand(
    mask_identity_t<float> mask, vector<float> arg, dx::zero_t zero) noexcept {
    auto const vzero = xmm::broadcast<int32>(zero);
    auto const kmask = _mm_cmp_epi32_mask(
        +xmm::reinterpret<int32>(mask), vzero, _MM_CMPINT_LT);
    return _mm_maskz_expand_ps(kmask, +arg);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int64> DPL_VECTORCALL expand(
    mask_identity_t<int64> mask, vector<int64> arg, dx::zero_t zero) noexcept {
    auto const vzero = xmm::broadcast<int64>(zero);
    auto const kmask = _mm_cmp_epi64_mask(+mask, vzero, _MM_CMPINT_LT);
    return _mm_maskz_expand_epi64(kmask, +arg);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32> DPL_VECTORCALL expand(
    mask_identity_t<int32> mask, vector<int32> arg, dx::zero_t zero) noexcept {
    auto const vzero = xmm::broadcast<int32>(zero);
    auto const kmask = _mm_cmp_epi32_mask(+mask, vzero, _MM_CMPINT_LT);
    return _mm_maskz_expand_epi32(kmask, +arg);
}

template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double>
    DPL_VECTORCALL expand(
        cmask_t<double, M>, vector<double> arg, dx::zero_t) noexcept {
    return _mm_maskz_expand_pd(M, +arg);
}

template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL expand(
        cmask_t<float, M>, vector<float> arg, dx::zero_t) noexcept {
    return _mm_maskz_expand_ps(M, +arg);
}

template <imask_t<int64> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int64>
    DPL_VECTORCALL expand(
        cmask_t<int64, M>, vector<int64> arg, dx::zero_t) noexcept {
    return _mm_maskz_expand_epi64(M, +arg);
}

template <imask_t<int32> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32>
    DPL_VECTORCALL expand(
        cmask_t<int32, M>, vector<int32> arg, dx::zero_t) noexcept {
    return _mm_maskz_expand_epi32(M, +arg);
}

#  endif

#  if DPL_SIMD_X86_AVX512VBMI2

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int16> DPL_VECTORCALL expand(mask_identity_t<int16> mask,
    vector<int16> arg, vector<int16> src) noexcept {
    auto const zero = xmm::broadcast<int16>(dx::zero);
    auto const kmask = _mm_cmp_epi16_mask(+mask, zero, _MM_CMPINT_LT);
    return _mm_mask_expand_epi16(+src, kmask, +arg);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int8> DPL_VECTORCALL expand(
    mask_identity_t<int8> mask, vector<int8> arg, vector<int8> src) noexcept {
    auto const zero = xmm::broadcast<int8>(dx::zero);
    auto const kmask = _mm_cmp_epi8_mask(+mask, zero, _MM_CMPINT_LT);
    return _mm_mask_expand_epi8(+src, kmask, +arg);
}

template <imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int16>
    DPL_VECTORCALL expand(
        cmask_t<int16, M>, vector<int16> arg, vector<int16> src) noexcept {
    return _mm_mask_expand_epi16(+src, M, +arg);
}

template <imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int8>
    DPL_VECTORCALL expand(
        cmask_t<int8, M>, vector<int8> arg, vector<int8> src) noexcept {
    return _mm_mask_expand_epi8(+src, M, +arg);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int16> DPL_VECTORCALL expand(
    mask_identity_t<int16> mask, vector<int16> arg, dx::zero_t zero) noexcept {
    auto const vzero = xmm::broadcast<int16>(zero);
    auto const kmask = _mm_cmp_epi16_mask(+mask, vzero, _MM_CMPINT_LT);
    return _mm_maskz_expand_epi16(kmask, +arg);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int8> DPL_VECTORCALL expand(
    mask_identity_t<int8> mask, vector<int8> arg, dx::zero_t zero) noexcept {
    auto const vzero = xmm::broadcast<int8>(zero);
    auto const kmask = _mm_cmp_epi8_mask(+mask, vzero, _MM_CMPINT_LT);
    return _mm_maskz_expand_epi8(kmask, +arg);
}

template <imask_t<int16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int16>
    DPL_VECTORCALL expand(
        cmask_t<int16, M>, vector<int16> arg, dx::zero_t) noexcept {
    return _mm_maskz_expand_epi16(M, +arg);
}

template <imask_t<int8> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int8>
    DPL_VECTORCALL expand(
        cmask_t<int8, M>, vector<int8> arg, dx::zero_t) noexcept {
    return _mm_maskz_expand_epi8(M, +arg);
}

#  endif

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> expand(
    mask_identity_t<E> mask, vector<E> arg, vector<E> src) noexcept
requires requires(
    vector<signed_representation_t<E>> arg) { xmm::expand(mask, arg, arg); }
{
    using sint_t = signed_representation_t<E>;
    return xmm::reinterpret<E>(xmm::expand(
        mask, xmm::reinterpret<sint_t>(arg), xmm::reinterpret<sint_t>(src)));
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> expand(
    cmask_t<E, M> mask, vector<E> arg, vector<E> src) noexcept
requires requires(
    vector<signed_representation_t<E>> arg) { xmm::expand(mask, arg, arg); }
{
    using sint_t = signed_representation_t<E>;
    return xmm::reinterpret<E>(xmm::expand(
        mask, xmm::reinterpret<sint_t>(arg), xmm::reinterpret<sint_t>(src)));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> expand(
    mask_identity_t<E> mask, vector<E> arg, dx::zero_t zero) noexcept
requires requires(
    vector<signed_representation_t<E>> arg) { xmm::expand(mask, arg, zero); }
{
    using sint_t = signed_representation_t<E>;
    return xmm::reinterpret<E>(
        xmm::expand(mask, xmm::reinterpret<sint_t>(arg), zero));
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> expand(
    cmask_t<E, M> mask, vector<E> arg, dx::zero_t zero) noexcept
requires requires(
    vector<signed_representation_t<E>> arg) { xmm::expand(mask, arg, zero); }
{
    using sint_t = signed_representation_t<E>;
    return xmm::reinterpret<E>(
        xmm::expand(mask, xmm::reinterpret<sint_t>(arg), zero));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> expand(
    abi_tag, mask_identity_t<E> mask, vector<E> arg, vector<E> src) noexcept
requires requires { xmm::expand(mask, arg, src); }
{
    return xmm::expand(mask, arg, src);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> expand(
    abi_tag, cmask_t<E, M> mask, vector<E> arg, vector<E> src) noexcept
requires requires { xmm::expand(mask, arg, src); }
{
    return xmm::expand(mask, arg, src);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> expand(
    abi_tag, mask_identity_t<E> mask, vector<E> arg, dx::zero_t zero) noexcept
requires requires { xmm::expand(mask, arg, zero); }
{
    return xmm::expand(mask, arg, zero);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> expand(
    abi_tag, cmask_t<E, M> mask, vector<E> arg, dx::zero_t zero) noexcept
requires requires { xmm::expand(mask, arg, zero); }
{
    return xmm::expand(mask, arg, zero);
}
}

__DPL_DEFAULT_NAMESPACE_END
#endif
