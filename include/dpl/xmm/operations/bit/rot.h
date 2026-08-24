// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  if !DPL_MODULES
#    include "dpl/core/immediate/immediate.h"
#    include "dpl/std/bit/char_bit.h"
#    include "dpl/std/type_traits/type_identity.h"
#    include "dpl/xmm/basic/abi.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
#  if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
template <integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32>
    DPL_VECTORCALL rotl(vector<int32> val, N) noexcept {
    return _mm_rol_epi32(+val, static_cast<int>(N::value));
}

template <integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint32>
    DPL_VECTORCALL rotl(vector<uint32> val, N) noexcept {
    return _mm_rol_epi32(+val, static_cast<int>(N::value));
}

template <integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int64>
    DPL_VECTORCALL rotl(vector<int64> val, N) noexcept {
    return _mm_rol_epi64(+val, static_cast<int>(N::value));
}

template <integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint64>
    DPL_VECTORCALL rotl(vector<uint64> val, N) noexcept {
    return _mm_rol_epi64(+val, static_cast<int>(N::value));
}

template <imask_t<int32> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32>
    DPL_VECTORCALL rotl(
        vector<int32> src, cmask_t<int32, M>, vector<int32> val, N) noexcept {
    return _mm_mask_rol_epi32(+src, M, +val, static_cast<int>(N::value));
}

template <imask_t<uint32> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint32>
    DPL_VECTORCALL rotl(vector<uint32> src, cmask_t<uint32, M>,
        vector<uint32> val, N) noexcept {
    return _mm_mask_rol_epi32(+src, M, +val, static_cast<int>(N::value));
}

template <imask_t<int64> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int64>
    DPL_VECTORCALL rotl(
        vector<int64> src, cmask_t<int64, M>, vector<int64> val, N) noexcept {
    return _mm_mask_rol_epi64(+src, M, +val, static_cast<int>(N::value));
}

template <imask_t<uint64> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint64>
    DPL_VECTORCALL rotl(vector<uint64> src, cmask_t<uint64, M>,
        vector<uint64> val, N) noexcept {
    return _mm_mask_rol_epi64(+src, M, +val, static_cast<int>(N::value));
}

template <imask_t<int32> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int32>
    DPL_VECTORCALL rotl(
        dx::zero_t, cmask_t<int32, M>, vector<int32> val, N) noexcept {
    return _mm_maskz_rol_epi32(M, +val, static_cast<int>(N::value));
}

template <imask_t<uint32> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint32>
    DPL_VECTORCALL rotl(
        dx::zero_t, cmask_t<uint32, M>, vector<uint32> val, N) noexcept {
    return _mm_maskz_rol_epi32(M, +val, static_cast<int>(N::value));
}

template <imask_t<int64> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<int64>
    DPL_VECTORCALL rotl(
        dx::zero_t, cmask_t<int64, M>, vector<int64> val, N) noexcept {
    return _mm_maskz_rol_epi64(M, +val, static_cast<int>(N::value));
}

template <imask_t<uint64> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint64>
    DPL_VECTORCALL rotl(
        dx::zero_t, cmask_t<uint64, M>, vector<uint64> val, N) noexcept {
    return _mm_maskz_rol_epi64(M, +val, static_cast<int>(N::value));
}

namespace details {
template <simd_element E, integral_constant_like N>
inline constexpr auto rshift =
    imm<__DPL type_bit_v<E> - static_cast<size_t>(N::value)>;

}

template <simd_element E, integral_constant_like N>
inline vector<E> DPL_VECTORCALL rotr(vector<E> val, N) noexcept
requires requires { xmm::rotl(val, details::rshift<E, N>); }
{
    return rotl(val, details::rshift<E, N>);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
inline vector<E> DPL_VECTORCALL rotr(type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> val, N) noexcept
requires requires { xmm::rotl(src, mask, val, details::rshift<E, N>); }
{
    return rotl(src, mask, val, details::rshift<E, N>);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
inline vector<E> DPL_VECTORCALL rotr(
    dx::zero_t zero, cmask_t<E, M> mask, vector<E> val, N) noexcept
requires requires { xmm::rotl(zero, mask, val, details::rshift<E, N>); }
{
    return rotl(zero, mask, val, details::rshift<E, N>);
}

template <simd_element E, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL rotl(abi_tag, vector<E> lhs, N rhs) noexcept
requires requires { xmm::rotl(lhs, rhs); }
{
    return xmm::rotl(lhs, rhs);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL rotl(abi_tag, type_identity_t<vector<E>> src,
        cmask_t<E, M> mask, vector<E> lhs, N rhs) noexcept
requires requires { xmm::rotl(src, mask, lhs, rhs); }
{
    return xmm::rotl(src, mask, lhs, rhs);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL rotl(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
        vector<E> lhs, N rhs) noexcept
requires requires { xmm::rotl(zero, mask, lhs, rhs); }
{
    return xmm::rotl(zero, mask, lhs, rhs);
}

template <simd_element E, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL rotr(abi_tag, vector<E> lhs, N rhs) noexcept
requires requires { xmm::rotr(lhs, rhs); }
{
    return xmm::rotr(lhs, rhs);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL rotr(abi_tag, type_identity_t<vector<E>> src,
        cmask_t<E, M> mask, vector<E> lhs, N rhs) noexcept
requires requires { xmm::rotr(src, mask, lhs, rhs); }
{
    return xmm::rotr(src, mask, lhs, rhs);
}

template <simd_element E, imask_t<E> M, integral_constant_like N>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL rotr(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
        vector<E> lhs, N rhs) noexcept
requires requires { xmm::rotr(zero, mask, lhs, rhs); }
{
    return xmm::rotr(zero, mask, lhs, rhs);
}
#  else  // if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
void rotl(abi_tag, ...) noexcept = delete;
void rotr(abi_tag, ...) noexcept = delete;
#  endif // if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
