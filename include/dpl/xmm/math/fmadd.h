// Copyright 2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_FMA

#  if !DPL_MODULES
#    include "dpl/std/type_traits/type_identity.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/operations/element_cast.h"
#    include "dpl/xmm/operations/reinterpret.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float> DPL_VECTORCALL fmadd(
    vector<float> lhs, vector<float> mid, vector<float> rhs) noexcept {
    return _mm_fmadd_ps(+lhs, +mid, +rhs);
}

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double> DPL_VECTORCALL fmadd(
    vector<double> lhs, vector<double> mid, vector<double> rhs) noexcept {
    return _mm_fmadd_pd(+lhs, +mid, +rhs);
}

#  if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::float16> DPL_VECTORCALL fmadd(vector<ext::float16> lhs,
    vector<ext::float16> mid, vector<ext::float16> rhs) noexcept {
    return _mm_fmadd_ph(+lhs, +mid, +rhs);
}
#  endif

DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::bfloat16> DPL_VECTORCALL fmadd(vector<ext::bfloat16> lhs,
    vector<ext::bfloat16> mid, vector<ext::bfloat16> rhs) noexcept {
    auto const hlhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+lhs), 8)));
    auto const hmid = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+mid), 8)));
    auto const hrhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+rhs), 8)));
    auto const lo = xmm::fmadd(xmm::element_cast<float>(lhs),
        xmm::element_cast<float>(mid), xmm::element_cast<float>(rhs));
    auto const hi = xmm::fmadd(xmm::element_cast<float>(hlhs),
        xmm::element_cast<float>(hmid), xmm::element_cast<float>(hrhs));
#  if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtne2ps_pbh(+hi, +lo);
#  else
    auto const packed = _mm_shuffle_pd(
        __DPL bit_cast<__m128d>(+xmm::element_cast<ext::bfloat16>(lo)),
        __DPL bit_cast<__m128d>(+xmm::element_cast<ext::bfloat16>(hi)), 0);
    return xmm::reinterpret<ext::bfloat16>(vector<double>(packed));
#  endif
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr vector<E> fmadd(
    abi_tag, vector<E> lhs, vector<E> mid, vector<E> rhs) noexcept
requires requires { xmm::fmadd(lhs, mid, rhs); }
{
    return xmm::fmadd(lhs, mid, rhs);
}

#  if DPL_SIMD_X86_AVX512VL
#    if DPL_SIMD_X86_AVX512F
template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL fmadd(vector<float> lhs, cmask_t<float, M> mask,
        vector<float> mid, vector<float> rhs) noexcept {
    return _mm_mask_fmadd_ps(+lhs, M, +mid, +rhs);
}

template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL fmadd(dx::zero_t, cmask_t<float, M> mask, vector<float> lhs,
        vector<float> mid, vector<float> rhs) noexcept {
    return _mm_maskz_fmadd_ps(M, +lhs, +mid, +rhs);
}

template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL fmadd(vector<float> lhs, vector<float> mid,
        vector<float> rhs, cmask_t<float, M> mask) noexcept {
    return _mm_mask3_fmadd_ps(+lhs, +mid, +rhs, M);
}

template <imask_t<ext::bfloat16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::bfloat16>
    DPL_VECTORCALL fmadd(vector<ext::bfloat16> lhs,
        cmask_t<ext::bfloat16, M> mask, vector<ext::bfloat16> mid,
        vector<ext::bfloat16> rhs) noexcept {

    auto const hlhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+lhs), 8)));
    auto const hmid = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+mid), 8)));
    auto const hrhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+rhs), 8)));
    constexpr cmask_t<float, (M & 0xf)> lmask;
    constexpr cmask_t<float, (M >> 4)> hmask;
    auto const lo = xmm::fmadd(xmm::element_cast<float>(lhs), lmask,
        xmm::element_cast<float>(mid), xmm::element_cast<float>(rhs));
    auto const hi = xmm::fmadd(xmm::element_cast<float>(hlhs), hmask,
        xmm::element_cast<float>(hmid), xmm::element_cast<float>(hrhs));

#      if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtne2ps_pbh(+hi, +lo);
#      else
    auto const packed = _mm_shuffle_pd(
        __DPL bit_cast<__m128d>(+xmm::element_cast<ext::bfloat16>(lo)),
        __DPL bit_cast<__m128d>(+xmm::element_cast<ext::bfloat16>(hi)), 0);
    return xmm::reinterpret<ext::bfloat16>(vector<double>(packed));
#      endif
}

template <imask_t<ext::bfloat16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::bfloat16>
    DPL_VECTORCALL fmadd(dx::zero_t zero, cmask_t<ext::bfloat16, M> mask,
        vector<ext::bfloat16> lhs, vector<ext::bfloat16> mid,
        vector<ext::bfloat16> rhs) noexcept {
    auto const hlhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+lhs), 8)));
    auto const hmid = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+mid), 8)));
    auto const hrhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+rhs), 8)));
    constexpr cmask_t<float, (M & 0xf)> lmask;
    constexpr cmask_t<float, (M >> 4)> hmask;
    auto const lo = xmm::fmadd(zero, lmask, xmm::element_cast<float>(lhs),
        xmm::element_cast<float>(mid), xmm::element_cast<float>(rhs));
    auto const hi = xmm::fmadd(zero, hmask, xmm::element_cast<float>(hlhs),
        xmm::element_cast<float>(hmid), xmm::element_cast<float>(hrhs));

#      if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtne2ps_pbh(+hi, +lo);
#      else
    auto const packed = _mm_shuffle_pd(
        __DPL bit_cast<__m128d>(+xmm::element_cast<ext::bfloat16>(lo)),
        __DPL bit_cast<__m128d>(+xmm::element_cast<ext::bfloat16>(hi)), 0);
    return xmm
#      endif
}

template <imask_t<ext::bfloat16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::bfloat16>
    DPL_VECTORCALL fmadd(vector<ext::bfloat16> lhs, vector<ext::bfloat16> mid,
        vector<ext::bfloat16> rhs, cmask_t<ext::bfloat16, M> mask) noexcept {
    auto const hlhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+lhs), 8)));
    auto const hmid = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+mid), 8)));
    auto const hrhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+rhs), 8)));
    constexpr cmask_t<float, (M & 0xf)> lmask;
    constexpr cmask_t<float, (M >> 4)> hmask;
    auto const lo = xmm::fmadd(xmm::element_cast<float>(lhs),
        xmm::element_cast<float>(mid), xmm::element_cast<float>(rhs), lmask);
    auto const hi = xmm::fmadd(xmm::element_cast<float>(hlhs),
        xmm::element_cast<float>(hmid), xmm::element_cast<float>(hrhs), hmask);

#      if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtne2ps_pbh(+hi, +lo);
#      else
    auto const packed = _mm_shuffle_pd(
        __DPL bit_cast<__m128d>(+xmm::element_cast<ext::bfloat16>(lo)),
        __DPL bit_cast<__m128d>(+xmm::element_cast<ext::bfloat16>(hi)), 0);
    return xmm
#      endif
}

template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double>
    DPL_VECTORCALL fmadd(vector<double> lhs, cmask_t<double, M> mask,
        vector<double> mid, vector<double> rhs) noexcept {
    return _mm_mask_fmadd_pd(+lhs, M, +mid, +rhs);
}

template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double>
    DPL_VECTORCALL fmadd(dx::zero_t, cmask_t<double, M> mask,
        vector<double> lhs, vector<double> mid, vector<double> rhs) noexcept {
    return _mm_maskz_fmadd_pd(M, +lhs, +mid, +rhs);
}
template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double>
    DPL_VECTORCALL fmadd(vector<double> lhs, vector<double> mid,
        vector<double> rhs, cmask_t<double, M> mask) noexcept {
    return _mm_mask3_fmadd_pd(+lhs, +mid, +rhs, M);
}
#    endif

#    if DPL_SIMD_X86_AVX512FP16
template <imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::float16>
    DPL_VECTORCALL fmadd(vector<ext::float16> lhs,
        cmask_t<ext::float16, M> mask, vector<ext::float16> mid,
        vector<ext::float16> rhs) noexcept {
    return _mm_mask_fmadd_ph(+lhs, M, +mid, +rhs);
}

template <imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::float16>
    DPL_VECTORCALL fmadd(dx::zero_t, cmask_t<ext::float16, M> mask,
        vector<ext::float16> lhs, vector<ext::float16> mid,
        vector<ext::float16> rhs) noexcept {
    return _mm_maskz_fmadd_ph(M, +lhs, +mid, +rhs);
}

template <imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::float16>
    DPL_VECTORCALL fmadd(vector<ext::float16> lhs, vector<ext::float16> mid,
        vector<ext::float16> rhs, cmask_t<ext::float16, M> mask) noexcept {
    return _mm_mask3_fmadd_ph(+lhs, +mid, +rhs, M);
}
#    endif

#  endif

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
