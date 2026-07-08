// Copyright 2025 Bryan Wong

#pragma once
#include "dpl/config.h"

// IWYU pragma: private, include "dpl/xmm/operations/arithmetic.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/arithmetic/cast.h"
#include "dpl/xmm/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/type_identity.h"
#  include "dpl/std/utility/template_barrier.h"
#  include "dpl/xmm/basic/abi.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

DPL_EXPORT template <template_barrier_t = template_barrier, simd_element E>
requires integral<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL add(vector<E> lhs, vector<E> rhs) noexcept {
    if constexpr (sizeof(E) == sizeof(int64)) {
        return _mm_add_epi64(+lhs, +rhs);
    } else if constexpr (sizeof(E) == sizeof(int32)) {
        return _mm_add_epi32(+lhs, +rhs);
    } else if constexpr (sizeof(E) == sizeof(int16)) {
        return _mm_add_epi16(+lhs, +rhs);
    } else {
        static_assert(sizeof(E) == sizeof(int8));
        return _mm_add_epi8(+lhs, +rhs);
    }
}

DPL_EXPORT DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL add(vector<float> lhs, vector<float> rhs) noexcept {
    return _mm_add_ps(+lhs, +rhs);
}

DPL_EXPORT DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double>
    DPL_VECTORCALL add(vector<double> lhs, vector<double> rhs) noexcept {
    return _mm_add_pd(+lhs, +rhs);
}

#if DPL_SIMD_X86_AVX512VL
#  if DPL_SIMD_X86_AVX512F

DPL_EXPORT template <imask_t<int32> M, integral E>
requires common_size_with<E, int32>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL add(type_identity_t<vector<E>> src, cmask_t<E, M>,
        vector<E> lhs, vector<E> rhs) noexcept {
    return _mm_mask_add_epi32(+src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<int32> M, integral E>
requires common_size_with<E, int32>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL add(
        dx::zero_t, cmask_t<E, M>, vector<E> lhs, vector<E> rhs) noexcept {
    return _mm_maskz_add_epi32(M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL add(vector<float> src, cmask_t<float, M>, vector<float> lhs,
        vector<float> rhs) noexcept {
    return _mm_mask_add_ps(+src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double>
    DPL_VECTORCALL add(vector<double> src, cmask_t<double, M>,
        vector<double> lhs, vector<double> rhs) noexcept {
    return _mm_mask_add_pd(+src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<float> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<float>
    DPL_VECTORCALL add(dx::zero_t, cmask_t<float, M>, vector<float> lhs,
        vector<float> rhs) noexcept {
    return _mm_maskz_add_ps(M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<double> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<double>
    DPL_VECTORCALL add(dx::zero_t, cmask_t<double, M>, vector<double> lhs,
        vector<double> rhs) noexcept {
    return _mm_maskz_add_pd(M, +lhs, +rhs);
}

#  endif

#  if DPL_SIMD_X86_AVX512DQ

DPL_EXPORT template <imask_t<int64> M, integral E>
requires common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL add(type_identity_t<vector<E>> src, cmask_t<E, M>,
        vector<E> lhs, vector<E> rhs) noexcept {
    return _mm_mask_add_epi64(+src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<int64> M, integral E>
requires common_size_with<E, int64>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL add(
        dx::zero_t, cmask_t<E, M>, vector<E> lhs, vector<E> rhs) noexcept {
    return _mm_maskz_add_epi64(M, +lhs, +rhs);
}

#  endif

#  if DPL_SIMD_X86_AVX512BW

DPL_EXPORT template <imask_t<int16> M, integral E>
requires common_size_with<E, int16>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL add(type_identity_t<vector<E>> src, cmask_t<E, M>,
        vector<E> lhs, vector<E> rhs) noexcept {
    return _mm_mask_add_epi16(+src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<int16> M, integral E>
requires common_size_with<E, int16>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL add(
        dx::zero_t, cmask_t<E, M>, vector<E> lhs, vector<E> rhs) noexcept {
    return _mm_maskz_add_epi16(M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<int8> M, integral E>
requires common_size_with<E, int8>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL add(type_identity_t<vector<E>> src, cmask_t<E, M>,
        vector<E> lhs, vector<E> rhs) noexcept {
    return _mm_mask_add_epi8(+src, M, +lhs, +rhs);
}
DPL_EXPORT template <imask_t<int8> M, integral E>
requires common_size_with<E, int8>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL add(
        dx::zero_t, cmask_t<E, M>, vector<E> lhs, vector<E> rhs) noexcept {
    return _mm_maskz_add_epi8(M, +lhs, +rhs);
}

#  endif

#  if DPL_SIMD_X86_AVX512FP16
DPL_EXPORT DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::float16>
    DPL_VECTORCALL add(
        vector<ext::float16> lhs, vector<ext::float16> rhs) noexcept {
    return _mm_add_ph(+lhs, +rhs);
}

DPL_EXPORT template <imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::float16>
    DPL_VECTORCALL add(vector<ext::float16> src, cmask_t<ext::float16, M>,
        vector<ext::float16> lhs, vector<ext::float16> rhs) noexcept {
    return _mm_mask_add_ph(src, M, +lhs, +rhs);
}

DPL_EXPORT template <imask_t<ext::float16> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::float16>
    DPL_VECTORCALL add(dx::zero_t zero, cmask_t<ext::float16, M>,
        vector<ext::float16> lhs, vector<ext::float16> rhs) noexcept {
    return _mm_maskz_add_ph(M, +lhs, +rhs);
}
#  endif
#endif

DPL_EXPORT template <same_as<ext::bfloat16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<ext::bfloat16>
    DPL_VECTORCALL add(vector<E> lhs, vector<E> rhs) noexcept {
    auto const hlhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+lhs), 8)));
    auto const hrhs = vector<ext::bfloat16>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+rhs), 8)));
    auto const lo = xmm::add(xmm::to_float(lhs), xmm::to_float(rhs));
    auto const hi = xmm::add(xmm::to_float(hlhs), xmm::to_float(hrhs));

#if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtne2ps_pbh(+hi, +lo);
#else
    auto const packed =
        _mm_packus_epi32( __DPL bit_cast<__m128i>(+xmm::element_cast<E>(lo)),
            __DPL bit_cast<__m128i>(+xmm::element_cast<E>(hi)));

    return xmm::reinterpret<ext::bfloat16>(vector<int16>(packed));
#endif
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL add(abi_tag, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::add(lhs, rhs); }
{
    return xmm::add(lhs, rhs);
}

DPL_EXPORT template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL add(abi_tag, type_identity_t<vector<E>> src,
        cmask_t<E, M> mask, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::add(src, mask, lhs, rhs); }
{
    return xmm::add(src, mask, lhs, rhs);
}

DPL_EXPORT template <simd_element E, imask_t<E> M>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL add(abi_tag, dx::zero_t zero, cmask_t<E, M> mask,
        vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::add(zero, mask, lhs, rhs); }
{
    return xmm::add(zero, mask, lhs, rhs);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
