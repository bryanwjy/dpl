// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/arithmetic_type.h"
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/constants/value_bits.h"
#  include "dpl/core/type_traits/common_arithmetic_type.h"
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/reinterpret.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <typename E>
using negated_type DPL_NODEBUG = common_arithmetic_type_t<E, E>;

DPL_EXPORT template <arithmetic_type E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<negated_type<E>> DPL_VECTORCALL
    abs(abi_tag tag, simd<E> val) noexcept {
    if constexpr (unsigned_integral<E> || requires {
                      requires enumeration<E> &&
                          unsigned_integral<underlying_type_t<E>>;
                  }) {
        return val;
    } else if constexpr (common_size_with<int64, E>) {
#if DPL_SIMD_X86_AVX512F & DPL_SIMD_X86_AVX512VL
        return _mm_abs_epi64(+val);
#else
        auto const vval = +val;
        auto const zero = _mm_setzero_si128();
        auto const sign = _mm_cmpgt_epi64(zero, vval);
        return _mm_sub_epi64(_mm_xor_si128(vval, sign), sign);
#endif
    } else if constexpr (common_size_with<int32, E>) {
        return _mm_abs_epi32(+val);
    } else if constexpr (common_size_with<int16, E>) {
        return _mm_abs_epi16(+val);
    } else {
        static_assert(common_size_with<int8, E>);
        return _mm_abs_epi8(+val);
    }
}

DPL_EXPORT template <arithmetic_type E>
requires floating_point<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<negated_type<E>> DPL_VECTORCALL
    abs(abi_tag tag, simd<E> val) noexcept {
    if constexpr (common_float_with<E, float>) {
        return _mm_andnot_ps(_mm_set1_ps(-0.0f), +val);
    } else if constexpr (common_float_with<E, double>) {
        return _mm_andnot_pd(_mm_set1_pd(-0.0), +val);
    } else if constexpr (brain_float<E>) {
        using sbit = signed_representation_t<E>;
        auto const vval = +xmm::reinterpret<sbit>(tag, val);
        simd<sbit> const result =
            _mm_and_si128(vval, _mm_set1_epi16(dx::value_bits));
        return xmm::reinterpret<E>(tag, result);
    } else {
        static_assert(sizeof(E) == 2);
#if DPL_SIMD_X86_AVX512FP16 & DPL_SIMD_X86_AVX512VL
        return _mm_abs_ph(+val);
#else
        using sbit = signed_representation_t<E>;
        auto const vval = +xmm::reinterpret<sbit>(tag, val);
        simd<sbit> const result =
            _mm_and_si128(vval, _mm_set1_epi16(dx::value_bits));
        return xmm::reinterpret<E>(tag, result);
#endif
    }
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
