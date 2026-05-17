// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/basic/immediate.h"
#  include "dpl/core/basic/to_simd_mask.h"
#  include "dpl/core/concepts/common_float_with.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/extract.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E>
    DPL_VECTORCALL to_simd_mask(abi_tag tag, simd<E> src) noexcept {
    if consteval {
        []<size_t... Is>(abi_tag tag, simd<E> src, index_sequence<Is...>) {
            using bit = signed_representation_t<E>;
            return xmm::initialize<E>(tag,
                (__DPL bit_cast<bit>(xmm::extract(tag, src, imm<Is>)) !=
                    0)...);
        }(tag, src, iota_sequence<E, abi_tag>);
    } else {
        auto const result = [](abi_tag tag, simd<E> src) {
            if constexpr (common_float_with<E, float>) {
                return _mm_cmpneq_ps(+src, _mm_setzero_ps());
            } else if constexpr (common_float_with<E, double>) {
                return _mm_cmpneq_pd(+src, _mm_setzero_pd());
            } else if constexpr (floating_point<E> && sizeof(E) == 2) {
                auto const vsrc = __DPL bit_cast<__m128i>(+src);
                auto const abs = _mm_and_si128(vsrc, _mm_set1_epi16(0x7fff));
                using sbit = signed_representation_t<E>;
                auto const mask = _mm_xor_si128(
                    _mm_cmpeq_epi16(zero, +src), _mm_set1_epi16(-1));
                return +xmm::reinterpret<E>(tag, simd<sbit>(mask));
            } else {
                auto const all = [](__m128i val) {
                    return _mm_cmpeq_epi32(val, val);
                }(_mm_undefined_si128());
                auto const zero = _mm_xor_si128(all, all);
                if constexpr (common_size_with<E, int64>) {
                    return _mm_xor_si128(_mm_cmpeq_epi64(zero, +src), all);
                } else if constexpr (common_size_with<E, int32>) {
                    return _mm_xor_si128(_mm_cmpeq_epi32(zero, +src), all);
                } else if constexpr (common_size_with<E, int16>) {
                    return _mm_xor_si128(_mm_cmpeq_epi16(zero, +src), all);
                } else {
                    static_assert(common_size_with<E, int8>);
                    return _mm_xor_si128(_mm_cmpeq_epi8(zero, +src), all);
                }
            }
        }(tag, src);
        return xmm::reinterpret<E>(tag, result);
    }
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E>
    DPL_VECTORCALL to_simd_mask(
        abi_tag tag, assume_normalized_mask_t, simd<E> src) noexcept {
    return +src;
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E>
    DPL_VECTORCALL to_simd_mask(simd<E> src) noexcept {
    return xmm::to_simd_mask(xmm::abi, src);
}

DPL_EXPORT template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E>
    DPL_VECTORCALL to_simd_mask(
        assume_normalized_mask_t tag, simd<E> src) noexcept {
    return xmm::to_simd_mask(xmm::abi, tag, src);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
