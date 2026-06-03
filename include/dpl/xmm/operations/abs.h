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
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/constants/value_bits.h"
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/xmm/basic/abi.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

DPL_EXPORT template <typename E>
requires arithmetic_type<lane_representation_t<E>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL abs(abi_tag tag, simd<E> val) noexcept {
    if constexpr (unsigned_integral<E>) {
        return val;
    } else if constexpr (sizeof(E) == sizeof(int64)) {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
        return _mm_abs_epi64(+val);
#else
        auto const vval = +val;
        auto const zero = _mm_setzero_si128();
        auto const sign = _mm_cmpgt_epi64(zero, vval);
        return _mm_sub_epi64(_mm_xor_si128(vval, sign), sign);
#endif
    } else if constexpr (sizeof(E) == sizeof(int32)) {
        return _mm_abs_epi32(+val);
    } else if constexpr (sizeof(E) == sizeof(int16)) {
        return _mm_abs_epi16(+val);
    } else {
        static_assert(sizeof(E) == sizeof(int8));
        return _mm_abs_epi8(+val);
    }
}

DPL_EXPORT template <typename E>
requires arithmetic_type<lane_representation_t<E>> &&
    (floating_point<lane_representation_t<E>> ||
        bfloat16_like<lane_representation_t<E>> ||
        float16_like<lane_representation_t<E>>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL abs(abi_tag, simd<E> val) noexcept {
    if constexpr (same_as<lane_representation_t<E>, float>) {
        return _mm_andnot_ps(_mm_set1_ps(-0.0f), +val);
    } else if constexpr (same_as<lane_representation_t<E>, double>) {
        return _mm_andnot_pd(_mm_set1_pd(-0.0), +val);
    } else if constexpr (bfloat16_like<lane_representation_t<E>>) {
        auto const vval = +xmm::reinterpret<int16>(val);
        simd<int16> const result =
            _mm_and_si128(vval, _mm_set1_epi16(dx::value_bits));
        return xmm::reinterpret<E>(result);
    } else {
        static_assert(float16_like<lane_representation_t<E>>);
#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
        return _mm_abs_ph(+val);
#else
        auto const vval = +xmm::reinterpret<int16>(val);
        simd<int16> const result =
            _mm_and_si128(vval, _mm_set1_epi16(dx::value_bits));
        return xmm::reinterpret<E>(result);
#endif
    }
}

DPL_EXPORT template <typename E>
requires arithmetic_type<lane_representation_t<E>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL abs(simd<E> val) noexcept
requires requires { xmm::abs(xmm::abi, val); }
{
    return xmm::abs(xmm::abi, val);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
