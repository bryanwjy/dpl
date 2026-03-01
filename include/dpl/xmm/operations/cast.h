

// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/bitwise.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/concepts/common_arithmetic_with.h"
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/common_integral_with.h"
#  include "dpl/core/constants/max_value.h"
#  include "dpl/core/constants/min_value.h"
#  include "dpl/core/constants/msb.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/reinterpret.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <common_float_with<double> To, common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return +src;
}

template <common_float_with<double> To, common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvtps_pd(+src);
}

#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
template <common_float_with<double> To, fp16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvtph_pd(+src);
}
#endif

#if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
template <common_float_with<double> To, common_integral_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvtepi64_pd(+src);
}
#endif

template <common_float_with<double> To, common_integral_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvtepi32_pd(+src);
}

template <common_float_with<float> To, common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvtpd_ps(+src);
}

template <common_float_with<float> To, common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return +src;
}

#if DPL_SIMD_X86_AVX2
template <common_float_with<float> To, fp16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvtph_ps(+src);
}
#endif

template <common_float_with<float> TE, brain_float FE>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<TE> DPL_VECTORCALL cast(abi_tag, simd<FE> from) noexcept {
#if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtpbh_ps(+from);
#else
    auto const src = __DPL bit_cast<__m128i>(+from);
    return _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(src), 16));
#endif
}

#if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
template <common_float_with<float> To, common_arithmetic_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu64_ps(+src);
    } else {
        return _mm_cvtepi64_ps(+src);
    }
}
#endif

template <common_float_with<float> To, common_integral_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvtepi32_ps(+src);
}

template <brain_float TE, common_float_with<float> FE>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<TE> DPL_VECTORCALL cast(abi_tag, simd<FE> from) noexcept {
#if DPL_SIMD_X86_AVX512BF16 & DPL_SIMD_X86_AVX512VL
    return _mm_cvtneps_pbh(+from);
#else
    auto const ival = _mm_castps_si128(+from);
    // round to nearest even
    auto const lsb = _mm_and_si128(_mm_srli_epi32(ival, 16), _mm_set1_epi32(1));
    auto const bias = _mm_add_epi32(_mm_set1_epi32(0x7fff), lsb);
    return __DPL bit_cast<__m128bh>(
        _mm_srli_epi32(_mm_add_epi32(ival, bias), 16));
#endif
}

template <brain_float To, arithmetic_type E>
requires (!common_float_with<E, float>) &&
    requires(abi_tag tag, simd<E> src) { xmm::cast<float>(tag, src); }
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
    return xmm::cast<To>(tag, xmm::cast<float>(tag, src));
}

template <arithmetic_type To, brain_float E>
requires (!common_float_with<To, float>) &&
    requires(abi_tag tag, simd<float> src) { xmm::cast<To>(tag, src); }
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
    return xmm::cast<To>(tag, xmm::cast<float>(tag, src));
}

#if DPL_SIMD_X86_AVX2
template <fp16_like To, common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvtps_ph(+src);
}
#endif

#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
template <fp16_like To, common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvtpd_ph(+src);
}

template <fp16_like To, common_arithmetic_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu64_ph(+src);
    } else {
        return _mm_cvtepi64_ph(+src);
    }
}

template <fp16_like To, common_arithmetic_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu32_ph(+src);
    } else {
        return _mm_cvtepi32_ph(+src);
    }
}

template <fp16_like To, common_arithmetic_with<int16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu16_ph(+src);
    } else {
        return _mm_cvtepi16_ph(+src);
    }
}

template <fp16_like To, common_integral_with<int8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
    using i16 = signed_representation_t<To>;
    return _mm_cvtepi16_ph(+xmm::cast<i16>(tag, src));
}
#endif

#if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL

template <common_arithmetic_with<int64> To, common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<To>) {
        return _mm_cvttpd_epu64(+src);
    } else {
        return _mm_cvttpd_epi64(+src);
    }
}

#endif

#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
template <common_arithmetic_with<int64> To, fp16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<To>) {
        return _mm_cvttph_epu64(+src);
    } else {
        return _mm_cvttph_epi64(+src);
    }
}
#endif

template <common_arithmetic_with<int64> To, common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512DQ & DPL_SIMD_X86_AVX512VL
    if constexpr (unsigned_integral<To>) {
        return _mm_cvttps_epu64(+src);
    } else {
        return _mm_cvttps_epi64(+src);
    }
#else
    auto const arg =
        +xmm::cast<uint64>(tag, xmm::reinterpret<uint32>(tag, src));
    constexpr auto mantissa_width = 23u;
    constexpr auto hidden_bit = 1u << mantissa_width;
    constexpr auto bias = 127;
    auto const bexp = _mm_and_si128(
        _mm_srli_epi32(arg, mantissa_width), _mm_set1_epi64x(0xff));
    auto const exp = _mm_sub_epi64(bexp, _mm_set1_epi64x(bias));
    auto const mantissa =
        _mm_or_si128(_mm_and_si128(arg, _mm_set1_epi64x(hidden_bit - 1)),
            _mm_set1_epi64x(hidden_bit));
    auto const mwidth = _mm_set1_epi64x(mantissa_width);
    auto const shift = _mm_abs_epi32(_mm_sub_epi32(exp, mwidth));
    static constexpr auto rshift = [](__m128i lhs, __m128i rhs) {
#  if DPL_SIMD_X86_AVX2
        return _mm_srlv_epi32(lhs, rhs);
#  else
        constexpr auto swap64 = _MM_SHUFFLE(1, 0, 3, 2);
        return _mm_unpacklo_epi64(_mm_srl_epi64(lhs, rhs),
            _mm_srl_epi64(_mm_shuffle_epi32(lhs, swap64),
                _mm_shuffle_epi32(rhs, swap64)));
#  endif
    };
    auto const large =
        +xmm::bwshift_left(tag, simd<E>(mantissa), simd<E>(shift));
    auto const small = rshift(mantissa, shift);
    auto const zero = _mm_setzero_si128();
    auto const result = _mm_andnot_si128(_mm_cmpgt_epi64(zero, exp),
        _mm_blendv_epi8(large, small, _mm_cmpgt_epi64(mwidth, exp)));
    if constexpr (signed_integral<To>) {
        constexpr auto maxs64 = static_cast<float>(dx::max_value_v<int64>);
        constexpr auto mins64 = static_cast<float>(dx::min_value_v<int64>);

        auto const nmask = +xmm::cast<uint64>(tag,
            xmm::reinterpret<uint32>(
                tag, simd<E>(_mm_cmplt_ps(+src, _mm_setzero_ps()))));
        auto const nresult =
            _mm_add_epi64(_mm_xor_si128(result, nmask), _mm_set1_epi64x(1));
        auto const invalid = +xmm::cast<uint64>(tag,
            xmm::reinterpret<uint32>(tag,
                simd<E>(_mm_or_ps(_mm_cmplt_ps(+src, mins64),
                    _mm_cmpgt_ps(+src, _mm_set1_ps(maxs64))))));

        return _mm_castpd_si128(_mm_blendv_pd(
            nresult, _mm_set1_epi64x(dx::msb), _mm_castsi128_pd(invalid)));
    } else {
        constexpr auto maxu64 = static_cast<float>(dx::max_value_v<uint64>);
        auto const invalid = +xmm::cast<uint64>(tag,
            xmm::reinterpret<uint32>(tag,
                simd<E>(_mm_or_ps(_mm_cmplt_ps(+src, _mm_setzero_ps()),
                    _mm_cmpgt_ps(+src, _mm_set1_ps(maxu64))))));
        return _mm_castpd_si128(_mm_blendv_pd(
            result, _mm_set1_pd(-0.0), _mm_castsi128_pd(invalid)));
    }
#endif
}

template <common_arithmetic_with<int64> To, common_arithmetic_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return +src;
}

template <common_arithmetic_with<int64> To, common_arithmetic_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu32_epi64(+src);
    } else {
        return _mm_cvtepi32_epi64(+src);
    }
}

template <common_arithmetic_with<int64> To, common_arithmetic_with<int16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu16_epi64(+src);
    } else {
        return _mm_cvtepi16_epi64(+src);
    }
}

template <common_arithmetic_with<int64> To, common_arithmetic_with<int8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu8_epi64(+src);
    } else {
        return _mm_cvtepi8_epi64(+src);
    }
}

template <common_integral_with<int32> To, common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvttpd_epi32(+src);
}

template <common_integral_with<int32> To, common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvttps_epi32(+src);
}

template <common_integral_with<uint32> To, common_float_with<float> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvttps_epu32(+src);
#else
    constexpr auto mantissa_width = 23u;
    constexpr auto hidden_bit = 1u << mantissa_width;
    constexpr auto bias = 127;

    auto const arg = +xmm::reinterpret<uint32>(tag, src);
    auto const bexp = _mm_and_si128(
        _mm_srli_epi32(arg, mantissa_width), _mm_set1_epi32(0xff));
    auto const exp = _mm_sub_epi32(bexp, _mm_set1_epi64x(bias));
    auto const mantissa =
        _mm_or_si128(_mm_and_si128(arg, _mm_set1_epi32(hidden_bit - 1)),
            _mm_set1_epi32(hidden_bit));
    auto const mwidth = _mm_set1_epi32(mantissa_width);
    auto const shift = _mm_sub_epi32(exp, mwidth);
    auto const large =
        +xmm::bwshift_left(tag, simd<E>(mantissa), simd<E>(shift));

    constexpr auto maxu32 = static_cast<float>(dx::max_value_v<uint32>);
    auto const invalid = +xmm::reinterpret<uint32>(tag,
        simd<E>(_mm_or_ps(_mm_cmplt_ps(+src, _mm_setzero_ps()),
            _mm_cmpgt_ps(+src, _mm_set1_ps(maxu32)))));
    auto const islarge = _mm_cmpgt_epi32(exp, mwidth);
    auto const result = _mm_blendv_ps(_mm_castsi128_ps(_mm_cvttps_epi32(+src)),
        _mm_castsi128_ps(large), _mm_castsi128_ps(islarge));
    return _mm_castps_si128(_mm_blendv_ps(result, _mm_set1_ps(-0.0f), invalid));
#endif
}

#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
template <common_integral_with<uint32> To, common_float_with<double> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvttpd_epu32(+src);
}
#endif

#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
template <common_integral_with<int32> To, fp16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvttph_epi32(+src);
}

template <common_integral_with<uint32> To, fp16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvttph_epu32(+src);
}
#endif

template <common_arithmetic_with<int32> To, common_arithmetic_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi64_epi32(+src);
#else
    return _mm_insert_epi64(
        _mm_shuffle_epi32(+src, _MM_SHUFFLE(2, 0, 2, 0)), 0, 1);
#endif
}

template <common_arithmetic_with<int32> To, common_arithmetic_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return +src;
}

template <common_arithmetic_with<int32> To, common_arithmetic_with<int16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu16_epi32(+src);
    } else {
        return _mm_cvtepi16_epi32(+src);
    }
}

template <common_arithmetic_with<int32> To, common_arithmetic_with<int8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu8_epi32(+src);
    } else {
        return _mm_cvtepi8_epi32(+src);
    }
}

template <common_arithmetic_with<int16> To, floating_point E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
    return xmm::cast<To>(tag, xmm::cast<int32>(tag, src));
}

#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
template <common_arithmetic_with<int16> To, fp16_like E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return _mm_cvttph_epi16(+src);
}
#endif

template <common_arithmetic_with<int16> To, common_arithmetic_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi64_epi16(+src);
#else
    // Extract low 32 bits of each 64-bit lane
    auto const dwords = _mm_shuffle_epi32(+src, _MM_SHUFFLE(2, 0, 2, 0));
    // Keep only low 16 bits
    auto const masked = _mm_and_si128(dwords, _mm_set1_epi32(0xffff));
    // Pack 32 → 16
    return _mm_packs_epi32(masked, _mm_setzero_si128());
#endif
}

template <common_arithmetic_with<int16> To, common_arithmetic_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi32_epi16(+src);
#else
    auto const masked = _mm_and_si128(+src, _mm_set1_epi32(0xffff));
    return _mm_packs_epi32(masked, _mm_setzero_si128());
#endif
}

template <common_arithmetic_with<int16> To, common_arithmetic_with<int16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return +src;
}

template <common_arithmetic_with<int16> To, common_arithmetic_with<int8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    if constexpr (unsigned_integral<E>) {
        return _mm_cvtepu8_epi16(+src);
    } else {
        return _mm_cvtepi8_epi16(+src);
    }
}

template <common_arithmetic_with<int8> To, floating_point E>
requires (!brain_float<E>) &&
    requires(abi_tag tag, simd<E> src) { xmm::cast<int16>(tag, src); }
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag tag, simd<E> src) noexcept {
    return xmm::cast<int8>(xmm::cast<int16>(tag, src));
}

template <common_arithmetic_with<int8> To, common_arithmetic_with<int8> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
    return +src;
}

template <common_arithmetic_with<int8> To, common_arithmetic_with<int16> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi16_epi8(+src);
#else
    // Keep only low 8 bits of each 16-bit element
    auto const masked = _mm_and_si128(+src, _mm_set1_epi16(0xff));
    // Pack 16-bit → 8-bit (safe now, no saturation possible)
    return _mm_packs_epi16(masked, _mm_setzero_si128());
#endif
}

template <common_arithmetic_with<int8> To, common_arithmetic_with<int32> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512BW & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi32_epi8(+src);
#else
    // Keep only low 8 bits of each dword
    auto const masked = _mm_and_si128(+src, _mm_set1_epi32(0xff));
    // 32 → 16
    auto const words = _mm_packs_epi32(masked, _mm_setzero_si128());
    // 16 → 8
    return _mm_packs_epi16(words, _mm_setzero_si128());
#endif
}

template <common_arithmetic_with<int8> To, common_arithmetic_with<int64> E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<To> DPL_VECTORCALL cast(abi_tag, simd<E> src) noexcept {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
    return _mm_cvtepi64_epi8(+src);
#else
    // Extract low 32 bits of each 64-bit lane
    auto const dwords = _mm_shuffle_epi32(+src, _MM_SHUFFLE(2, 0, 2, 0));
    // Keep only low 8 bits
    auto const masked = _mm_and_si128(dwords, _mm_set1_epi32(0xff));
    // 32 → 16
    auto const words = _mm_packs_epi32(masked, _mm_setzero_si128());
    // 16 → 8
    return _mm_packs_epi16(words, _mm_setzero_si128());
#endif
}
} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
