// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: private, include "dpl/xmm/operations/arithmetic.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/arithmetic/fwd.h"

#include "dpl/xmm/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/std/type_traits/type_identity.h"
#  include "dpl/xmm/basic/abi.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

DPL_EXPORT DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<float>
    DPL_VECTORCALL divide(simd<float> lhs, simd<float> rhs) noexcept {
    return _mm_div_ps(+lhs, +rhs);
}

DPL_EXPORT DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<double>
    DPL_VECTORCALL divide(simd<double> lhs, simd<double> rhs) noexcept {
    return _mm_div_pd(+lhs, +rhs);
}

#if DPL_SIMD_X86_AVX512VL
#  if DPL_SIMD_X86_AVX512F

DPL_EXPORT template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<float>
    DPL_VECTORCALL divide(simd<float> src, cmask_t<float, M>, simd<float> lhs,
        simd<float> rhs) noexcept {
    return _mm_mask_div_ps(+src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<double>
    DPL_VECTORCALL divide(simd<double> src, cmask_t<double, M>,
        simd<double> lhs, simd<double> rhs) noexcept {
    return _mm_mask_div_pd(+src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<float>
    DPL_VECTORCALL divide(dx::zero_t, cmask_t<float, M>, simd<float> lhs,
        simd<float> rhs) noexcept {
    return _mm_maskz_div_ps(M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<double>
    DPL_VECTORCALL divide(dx::zero_t, cmask_t<double, M>, simd<double> lhs,
        simd<double> rhs) noexcept {
    return _mm_maskz_div_pd(M, +lhs, +rhs);
}
#  endif

#  if DPL_SIMD_X86_AVX512FP16
DPL_EXPORT template <simd_element E>
requires float16_like<representation_t<E>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL divide(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_div_ph(+lhs, +rhs);
}

DPL_EXPORT template <imask_t<int16> M, simd_element E>
requires float16_like<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL divide(type_identity_t<simd<E>> src, cmask_t<E, M>,
        simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_mask_div_ph(src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<int16> M, simd_element E>
requires float16_like<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL divide(
        dx::zero_t zero, cmask_t<E, M>, simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_maskz_div_ph(M, +lhs, +rhs);
}
#  endif
#endif

DPL_EXPORT template <simd_element E>
requires bfloat16_like<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL divide(simd<E> lhs, simd<E> rhs) noexcept {
    auto const hlhs = simd<E>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+lhs), 8)));
    auto const hrhs = simd<E>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+rhs), 8)));
    auto const lo = xmm::divide(
        xmm::element_cast<float>(lhs), xmm::element_cast<float>(rhs));
    auto const hi = xmm::divide(
        xmm::element_cast<float>(hlhs), xmm::element_cast<float>(hrhs));

#if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtne2ps_pbh(hi, lo);
#else
    using sbit = signed_representation_t<E>;
    auto const packed =
        _mm_packus_epi32(__DPL bit_cast<__m128i>(+xmm::element_cast<E>(lo)),
            __DPL bit_cast<__m128i>(+xmm::element_cast<E>(hi)));

    return xmm::reinterpret<E>(simd<sbit>(packed));
#endif
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL divide(abi_tag, simd<E> lhs, simd<E> rhs) noexcept
requires requires { xmm::divide(lhs, rhs); }
{
    return xmm::divide(lhs, rhs);
}

DPL_EXPORT template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline simd<E>
    DPL_VECTORCALL divide(abi_tag, type_identity_t<simd<E>> src,
        cmask_t<E, M> mask, simd<E> lhs, simd<E> rhs) noexcept
requires requires { xmm::divide(src, mask, lhs, rhs); }
{
    return xmm::divide(src, mask, lhs, rhs);
}

DPL_EXPORT template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline simd<E>
    DPL_VECTORCALL divide(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
        simd<E> lhs, simd<E> rhs) noexcept
requires requires { xmm::divide(zero, mask, lhs, rhs); }
{
    return xmm::divide(zero, mask, lhs, rhs);
}
} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
