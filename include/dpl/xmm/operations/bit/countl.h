// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/operations/bit/popcount.h"
#  include "dpl/xmm/operations/reinterpret.h"

#  if !DPL_MODULES
#    include "dpl/core/fwd.h"

#    include "dpl/std/bit/countl.h"
#    include "dpl/xmm/basic/abi.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <common_size_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<unsigned_representation_t<E>>
    DPL_VECTORCALL countl_zero(vector<E> val) noexcept {
#  if DPL_SIMD_X86_AVX512CD && DPL_SIMD_X86_AVX512VL
    return _mm_lzcnt_epi64(+xmm::reinterpret<signed_representation_t<E>>(val));
#  else
    using sint = signed_representation_t<E>;
    auto const vval = xmm::reinterpret<sint>(val);
    return xmm::initialize<sint>( __DPL countl_zero(xmm::extract<0>(vval)),
        __DPL countl_zero(xmm::extract<1>(vval)));
#  endif
}

template <common_size_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<unsigned_representation_t<E>>
    DPL_VECTORCALL countl_zero(vector<E> val) noexcept {
#  if DPL_SIMD_X86_AVX512CD && DPL_SIMD_X86_AVX512VL
    return _mm_lzcnt_epi32(+xmm::reinterpret<signed_representation_t<E>>(val));
#  else
    using sint = signed_representation_t<E>;
    auto const vval = xmm::reinterpret<sint>(val);
    return xmm::initialize<sint>( __DPL countl_zero(xmm::extract<0>(vval)),
        __DPL countl_zero(xmm::extract<1>(vval)),
        __DPL countl_zero(xmm::extract<3>(vval)),
        __DPL countl_zero(xmm::extract<4>(vval)));
#  endif
}

#  if DPL_SIMD_X86_AVX512CD && DPL_SIMD_X86_AVX512VL
template <imask_t<uint64> M, common_size_with<uint64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint64>
    DPL_VECTORCALL countl_zero(
        vector<uint64> src, cmask_t<E, M>, vector<E> val) noexcept {
    return _mm_mask_lzcnt_epi64(+src, M, +xmm::reinterpret<uint64>(val));
}

template <imask_t<uint64> M, common_size_with<uint64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint64>
    DPL_VECTORCALL countl_zero(
        dx::zero_t, cmask_t<E, M>, vector<E> val) noexcept {
    return _mm_maskz_lzcnt_epi64(M, +xmm::reinterpret<uint64>(val));
}

template <imask_t<uint32> M, common_size_with<uint32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint32>
    DPL_VECTORCALL countl_zero(
        vector<uint32> src, cmask_t<E, M>, vector<E> val) noexcept {
    return _mm_mask_lzcnt_epi32(+src, M, +xmm::reinterpret<uint32>(val));
}

template <imask_t<uint32> M, common_size_with<uint32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint32>
    DPL_VECTORCALL countl_zero(
        dx::zero_t, cmask_t<E, M>, vector<E> val) noexcept {
    return _mm_maskz_lzcnt_epi32(M, +xmm::reinterpret<uint32>(val));
}
#  endif

template <common_size_with<uint16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint16>
    DPL_VECTORCALL countl_zero(vector<E> val) noexcept {
    auto vval = +xmm::reinterpret<uint16>(val);
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 1));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 2));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 4));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 8));
    return _mm_sub_epi16(_mm_set1_epi16(16), xmm::popcount(vval));
}

template <common_size_with<uint8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<uint8>
    DPL_VECTORCALL countl_zero(vector<E> val) noexcept {
    auto vval = +xmm::reinterpret<uint8>(val);
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 1));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 2));
    vval = _mm_or_si128(vval, _mm_srli_epi16(vval, 4));
    return _mm_sub_epi8(_mm_set1_epi8(8), xmm::popcount(vval));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<unsigned_representation_t<E>>
    DPL_VECTORCALL countl_one(vector<E> val) noexcept {
    using sbit = signed_representation_t<E>;
    auto vval = +xmm::reinterpret<sbit>(val);
    return xmm::countl_zero(
        vector<sbit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<unsigned_representation_t<E>>
    DPL_VECTORCALL countl_one(vector<unsigned_representation_t<E>> src,
        cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::countl_zero(src, mask, src); }
{
    using ubit = unsigned_representation_t<E>;
    auto vval = +xmm::reinterpret<ubit>(val);
    return xmm::countl_zero(src, mask,
        vector<ubit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<unsigned_representation_t<E>>
    DPL_VECTORCALL countl_one(
        dx::zero_t zero, cmask_t<E, M> mask, vector<E> val) noexcept
requires requires(vector<unsigned_representation_t<E>> rep) {
    xmm::countl_zero(zero, mask, rep);
}
{
    using ubit = unsigned_representation_t<E>;
    auto vval = +xmm::reinterpret<ubit>(val);
    return xmm::countl_zero(zero, mask,
        vector<ubit>(_mm_xor_si128(vval, _mm_cmpeq_epi32(vval, vval))));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countl_zero(abi_tag, vector<E> val) noexcept
requires requires { xmm::countl_zero(val); }
{
    return xmm::countl_zero(val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countl_zero(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::countl_zero(src, mask, val); }
{
    return xmm::countl_zero(src, mask, val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countl_zero(
    abi_tag, dx::zero_t zero, cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::countl_zero(zero, mask, val); }
{
    return xmm::countl_zero(zero, mask, val);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countl_one(abi_tag, vector<E> val) noexcept
requires requires { xmm::countl_one(val); }
{
    return xmm::countl_one(val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countl_one(abi_tag, type_identity_t<vector<E>> src,
    cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::countl_one(src, mask, val); }
{
    return xmm::countl_one(src, mask, val);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL countl_one(
    abi_tag, dx::zero_t zero, cmask_t<E, M> mask, vector<E> val) noexcept
requires requires { xmm::countl_one(zero, mask, val); }
{
    return xmm::countl_one(zero, mask, val);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
