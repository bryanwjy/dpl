// Copyright 2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2
#  include "dpl/xmm/operations/reinterpret.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/msb.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/broadcast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

inline mask<float> DPL_VECTORCALL cmpunord(
    vector<float> lhs, vector<float> rhs) noexcept {
    return _mm_cmpunord_ps(+lhs, +rhs);
    // return _mm_setzero_ps();
}

inline mask<double> DPL_VECTORCALL cmpunord(
    vector<double> lhs, vector<double> rhs) noexcept {
    return _mm_cmpunord_pd(+lhs, +rhs);
}

#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL

inline mask<ext::float16> DPL_VECTORCALL cmpunord(
    vector<ext::float16> lhs, vector<ext::float16> rhs) noexcept {
    return _mm_movm_epi16(_mm_cmp_ph_mask(+lhs, +rhs, _CMP_UNORD_Q));
}

template <imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<ext::float16>
    DPL_VECTORCALL cmpunord(cmask_t<ext::float16, M>, vector<ext::float16> lhs,
        vector<ext::float16> rhs) noexcept {
    return _mm_movm_epi16(_mm_mask_cmp_ph_mask(M, +lhs, +rhs, _CMP_UNORD_Q));
}

#  else

inline mask<ext::float16> DPL_VECTORCALL cmpunord(
    vector<ext::float16> lhs, vector<ext::float16> rhs) noexcept {
    using traits DPL_NODEBUG = floating_point_traits<ext::float16>;
    constexpr auto bound = __DPL bit_cast<int16>(traits::exponent_mask);
    auto const vbound = +xmm::broadcast<int16>(bound);
    auto const vsign = +xmm::broadcast<int16>(dx::msb);

    auto const ilhs = _mm_andnot_si128(vsign, +xmm::reinterpret<int16>(lhs));
    auto const irhs = _mm_andnot_si128(vsign, +xmm::reinterpret<int16>(rhs));
    return _mm_or_si128(
        _mm_cmpgt_epi16(ilhs, vbound), _mm_cmpgt_epi16(irhs, vbound));
}

#  endif

inline mask<ext::bfloat16> DPL_VECTORCALL cmpunord(
    vector<ext::bfloat16> lhs, vector<ext::bfloat16> rhs) noexcept {
    using traits DPL_NODEBUG = floating_point_traits<ext::bfloat16>;
    constexpr auto bound = __DPL bit_cast<int16>(traits::exponent_mask);
    auto const vbound = +xmm::broadcast<int16>(bound);
    auto const vsign = +xmm::broadcast<int16>(dx::msb);

    auto const ilhs = _mm_andnot_si128(vsign, +xmm::reinterpret<int16>(lhs));
    auto const irhs = _mm_andnot_si128(vsign, +xmm::reinterpret<int16>(rhs));
    return _mm_or_si128(
        _mm_cmpgt_epi16(ilhs, vbound), _mm_cmpgt_epi16(irhs, vbound));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> cmpunord(abi_tag, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::cmpunord(lhs, rhs); }
{
    return xmm::cmpunord(lhs, rhs);
}

template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline mask<E> cmpunord(
    abi_tag, cmask_t<E, M> mask, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::cmpunord(mask, lhs, rhs); }
{
    return xmm::cmpunord(mask, lhs, rhs);
}
} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
