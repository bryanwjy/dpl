// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/type_traits/common_arithmetic_type.h"
#  include "dpl/xmm/basic/abi.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <typename>
void element_cast(...) noexcept = delete;

template <arithmetic_type E>
using arithmetic_result DPL_NODEBUG = simd<common_arithmetic_type_t<E, E>>;

DPL_EXPORT template <arithmetic_type E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline arithmetic_result<E>
    DPL_VECTORCALL add(simd<E> lhs, simd<E> rhs) noexcept {
    if constexpr (common_arithmetic_with<E, int64>) {
        return _mm_add_epi64(+lhs, +rhs);
    } else if constexpr (common_arithmetic_with<E, int32>) {
        return _mm_add_epi32(+lhs, +rhs);
    } else if constexpr (common_arithmetic_with<E, int16>) {
        return _mm_add_epi16(+lhs, +rhs);
    } else {
        static_assert(common_arithmetic_with<E, int8>);
        return _mm_add_epi8(+lhs, +rhs);
    }
}

DPL_EXPORT template <common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL add(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_add_ps(+lhs, +rhs);
}

DPL_EXPORT template <common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL add(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_add_pd(+lhs, +rhs);
}

DPL_EXPORT template <arithmetic_type E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline arithmetic_result<E>
    DPL_VECTORCALL subtract(simd<E> lhs, simd<E> rhs) noexcept {
    if constexpr (common_arithmetic_with<E, int64>) {
        return _mm_sub_epi64(+lhs, +rhs);
    } else if constexpr (common_arithmetic_with<E, int32>) {
        return _mm_sub_epi32(+lhs, +rhs);
    } else if constexpr (common_arithmetic_with<E, int16>) {
        return _mm_sub_epi16(+lhs, +rhs);
    } else {
        static_assert(common_arithmetic_with<E, int8>);
        return _mm_sub_epi8(+lhs, +rhs);
    }
}

DPL_EXPORT template <common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL subtract(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_sub_ps(+lhs, +rhs);
}

DPL_EXPORT template <common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL subtract(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_sub_pd(+lhs, +rhs);
}

DPL_EXPORT template <arithmetic_type E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline arithmetic_result<E>
    DPL_VECTORCALL multiply(simd<E> lhs, simd<E> rhs) noexcept {
    if constexpr (common_arithmetic_with<E, int64>) {
#if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
        return _mm_mullo_epi64(+lhs, +rhs);
#else
        auto const vlhs = +lhs;
        auto const vrhs = +rhs;
        auto const left_lo = vlhs;
        auto const right_lo = vrhs;
        auto const left_hi = _mm_srli_epi64(vlhs, 32);
        auto const right_hi = _mm_srli_epi64(vrhs, 32);
        auto const lo_lo = _mm_mul_epu32(left_lo, right_lo);
        auto const lo_hi = _mm_mul_epu32(left_lo, right_hi);
        auto const hi_lo = _mm_mul_epu32(left_hi, right_lo);
        return _mm_add_epi64(
            lo_lo, _mm_slli_epi64(_mm_add_epi64(lo_hi, hi_lo), 32));
#endif
    } else if constexpr (common_arithmetic_with<E, int32>) {
        return _mm_mullo_epi32(+lhs, +rhs);
    } else if constexpr (common_arithmetic_with<E, int16>) {
        return _mm_mullo_epi16(+lhs, +rhs);
    } else {
        static_assert(common_arithmetic_with<E, int8>);
        auto const zero = _mm_setzero_si128();
        auto const vlhs = +lhs;
        auto const vrhs = +rhs;
        static constexpr auto is_unsigned = unsigned_integral<E> || requires {
            requires enumeration<E> && unsigned_integral<underlying_type_t<E>>;
        };
        static constexpr auto unpacklo = [](__m128i val, __m128i zero) {
            if constexpr (is_unsigned) {
                return _mm_unpacklo_epi8(val, zero);
            } else {
                return _mm_unpacklo_epi8(
                    val, _mm_cmpgt_epi8(_mm_setzero_si128(), val));
            }
        };
        static constexpr auto unpackhi = [](__m128i val, __m128i zero) {
            if constexpr (is_unsigned) {
                return _mm_unpackhi_epi8(val, zero);
            } else {
                return _mm_unpackhi_epi8(
                    val, _mm_cmpgt_epi8(_mm_setzero_si128(), val));
            }
        };
        auto const left_lo = unpacklo(vlhs, zero);
        auto const left_hi = unpackhi(vlhs, zero);
        auto const right_lo = unpacklo(vrhs, zero);
        auto const right_hi = unpackhi(vrhs, zero);

        auto const prod_lo = _mm_mullo_epi16(left_lo, right_lo);
        auto const prod_hi = _mm_mullo_epi16(left_hi, right_hi);
        return _mm_packus_epi16(prod_lo, prod_hi);
    }
}

DPL_EXPORT template <arithmetic_type E>
requires common_float_with<E, float>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_mul_ps(+lhs, +rhs);
}

DPL_EXPORT template <arithmetic_type E>
requires common_float_with<E, double>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_mul_pd(+lhs, +rhs);
}

DPL_EXPORT template <common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL divide(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_div_ps(+lhs, +rhs);
}

DPL_EXPORT template <common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL divide(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_div_pd(+lhs, +rhs);
}

#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
DPL_EXPORT template <float16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL add(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_add_ph(+lhs, +rhs);
}

DPL_EXPORT template <float16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL subtract(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_sub_ph(+lhs, +rhs);
}

DPL_EXPORT template <float16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_mul_ph(+lhs, +rhs);
}

DPL_EXPORT template <float16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL divide(simd<E> lhs, simd<E> rhs) noexcept {
    return _mm_div_ph(+lhs, +rhs);
}
#endif

DPL_EXPORT template <bfloat16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL add(simd<E> lhs, simd<E> rhs) noexcept {
    using f32 DPL_NODEBUG = details::front_t<float, E>;
    auto const hlhs = simd<E>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+lhs), 8)));
    auto const hrhs = simd<E>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+rhs), 8)));
    auto const lo =
        xmm::add(xmm::element_cast<f32>(lhs), xmm::element_cast<f32>(rhs));
    auto const hi =
        xmm::add(xmm::element_cast<f32>(hlhs), xmm::element_cast<f32>(hrhs));

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

DPL_EXPORT template <bfloat16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL subtract(simd<E> lhs, simd<E> rhs) noexcept {
    using f32 DPL_NODEBUG = details::front_t<float, E>;
    auto const hlhs = simd<E>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+lhs), 8)));
    auto const hrhs = simd<E>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+rhs), 8)));
    auto const lo =
        xmm::subtract(xmm::element_cast<f32>(lhs), xmm::element_cast<f32>(rhs));
    auto const hi = xmm::subtract(
        xmm::element_cast<f32>(hlhs), xmm::element_cast<f32>(hrhs));

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

DPL_EXPORT template <bfloat16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL multiply(simd<E> lhs, simd<E> rhs) noexcept {

    using f32 DPL_NODEBUG = details::front_t<float, E>;
    auto const hlhs = simd<E>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+lhs), 8)));
    auto const hrhs = simd<E>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+rhs), 8)));
    auto const lo =
        xmm::multiply(xmm::element_cast<f32>(lhs), xmm::element_cast<f32>(rhs));
    auto const hi = xmm::multiply(
        xmm::element_cast<f32>(hlhs), xmm::element_cast<f32>(hrhs));

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

DPL_EXPORT template <bfloat16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL divide(simd<E> lhs, simd<E> rhs) noexcept {
    using f32 DPL_NODEBUG = details::front_t<float, E>;
    auto const hlhs = simd<E>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+lhs), 8)));
    auto const hrhs = simd<E>(__DPL bit_cast<__m128bh>(
        _mm_srli_si128(__DPL bit_cast<__m128i>(+rhs), 8)));
    auto const lo =
        xmm::divide(xmm::element_cast<f32>(lhs), xmm::element_cast<f32>(rhs));
    auto const hi =
        xmm::divide(xmm::element_cast<f32>(hlhs), xmm::element_cast<f32>(hrhs));

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

DPL_EXPORT template <arithmetic_type E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL add(abi_tag, simd<E> lhs, simd<E> rhs) noexcept
requires requires { xmm::add(lhs, rhs); }
{
    return xmm::add(lhs, rhs);
}

DPL_EXPORT template <arithmetic_type E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL subtract(abi_tag, simd<E> lhs, simd<E> rhs) noexcept
requires requires { xmm::subtract(lhs, rhs); }
{
    return xmm::subtract(lhs, rhs);
}

DPL_EXPORT template <arithmetic_type E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL multiply(abi_tag, simd<E> lhs, simd<E> rhs) noexcept
requires requires { xmm::multiply(lhs, rhs); }
{
    return xmm::multiply(lhs, rhs);
}

DPL_EXPORT template <arithmetic_type E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL divide(abi_tag, simd<E> lhs, simd<E> rhs) noexcept
requires requires { xmm::divide(lhs, rhs); }
{
    return xmm::divide(lhs, rhs);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
