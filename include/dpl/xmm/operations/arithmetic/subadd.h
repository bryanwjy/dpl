// Copyright 2025 Bryan Wong

#pragma once
#include "dpl/config.h"

#if DPL_SIMD_X86_SSE3

#  include "dpl/xmm/operations/arithmetic/to_float.h"
#  include "dpl/xmm/operations/reinterpret.h"

#  if !DPL_MODULES
#    include "dpl/std/type_traits/type_identity.h"
#    include "dpl/std/utility/template_barrier.h"
#    include "dpl/xmm/basic/abi.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float> DPL_VECTORCALL subadd(
    vector<float> lhs, vector<float> rhs) noexcept {
    return _mm_addsub_ps(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double> DPL_VECTORCALL subadd(
    vector<double> lhs, vector<double> rhs) noexcept {
    return _mm_addsub_pd(+lhs, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::bfloat16> DPL_VECTORCALL subadd(
    vector<ext::bfloat16> lhs, vector<ext::bfloat16> rhs) noexcept {
    auto const hlhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+lhs), 8)));
    auto const hrhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+rhs), 8)));
    auto const lo = xmm::subadd(xmm::to_float(lhs), xmm::to_float(rhs));
    auto const hi = xmm::subadd(xmm::to_float(hlhs), xmm::to_float(hrhs));

#  if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtne2ps_pbh(+hi, +lo);
#  else
    auto const packed =
        _mm_shuffle_pd( __DPL bit_cast<__m128d>(+xmm::to_bfloat16(lo)),
            __DPL bit_cast<__m128d>(+xmm::to_bfloat16(hi)), 0);
    return xmm::reinterpret<ext::bfloat16>(vector<double>(packed));
#  endif
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL subadd(
    abi_tag, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::subadd(lhs, rhs); }
{
    return xmm::subadd(lhs, rhs);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL subadd(abi_tag, type_identity_t<vector<E>> src,
        cmask_t<E, M> mask, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::subadd(src, mask, lhs, rhs); }
{
    return xmm::subadd(src, mask, lhs, rhs);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL subadd(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
        vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::subadd(zero, mask, lhs, rhs); }
{
    return xmm::subadd(zero, mask, lhs, rhs);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
