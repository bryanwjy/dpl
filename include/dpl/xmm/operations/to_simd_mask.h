// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/operations/reinterpret.h"

#  if !DPL_MODULES
#    include "dpl/core/concepts/common_size_with.h"
#    include "dpl/core/immediate/immediate.h"
#    include "dpl/core/operations/to_simd_mask.h"
#    include "dpl/xmm/basic/abi.h"
#    include "dpl/xmm/basic/extract.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E>
    DPL_VECTORCALL to_simd_mask(abi_tag tag, vector<E> src) noexcept {
    if consteval {
        []<size_t... Is>(abi_tag tag, vector<E> src, index_sequence<Is...>) {
            using bit = signed_representation_t<E>;
            constexpr auto width = abi_tag::size / sizeof(E);
            using bitset_t = bitset<width>;
            auto const bits = bitset<width>((__DPL bit_cast<bit>(xmm::extract(
                                                 tag, src, imm<Is>)) != 0)...);
            return xmm::initialize<E>(tag, bits);
        }(tag, src, iota<E>);
    } else {
        auto const result = [](abi_tag tag, vector<E> src) {
            if constexpr (is_same_v<E, float>) {
                return _mm_cmpneq_ps(+src, _mm_setzero_ps());
            } else if constexpr (is_same_v<E, double>) {
                return _mm_cmpneq_pd(+src, _mm_setzero_pd());
            } else if constexpr (is_same_v<E, ext::float16> ||
                is_same_v<E, ext::bfloat16>) {
                auto const vsrc = __DPL bit_cast<__m128i>(+src);
                auto const abs =
                    _mm_and_si128(vsrc, +xmm::broadcast<int16>(0x7fff));
                using sbit = signed_representation_t<E>;
                auto const mask = _mm_xor_si128(
                    _mm_cmpeq_epi16(zero, +src), +xmm::broadcast<int16>(-1));
                return +xmm::reinterpret<E>(tag, vector<sbit>(mask));
            } else {
                auto const all = [](__m128i val) {
                    return _mm_cmpeq_epi32(val, val);
                }(_mm_undefined_si128());
                auto const zero = _mm_xor_si128(all, all);
                if constexpr (sizeof(E) == sizeof(int64)) {
                    return _mm_xor_si128(_mm_cmpeq_epi64(zero, +src), all);
                } else if constexpr (sizeof(E) == sizeof(int32)) {
                    return _mm_xor_si128(_mm_cmpeq_epi32(zero, +src), all);
                } else if constexpr (sizeof(E) == sizeof(int16)) {
                    return _mm_xor_si128(_mm_cmpeq_epi16(zero, +src), all);
                } else {
                    static_assert(sizeof(E) == sizeof(int8));
                    return _mm_xor_si128(_mm_cmpeq_epi8(zero, +src), all);
                }
            }
        }(tag, src);
        return xmm::reinterpret<E>(tag, result);
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E>
    DPL_VECTORCALL to_simd_mask(
        abi_tag tag, assume_normalized_mask_t, vector<E> src) noexcept {
    return +src;
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E>
    DPL_VECTORCALL to_simd_mask(vector<E> src) noexcept {
    return xmm::to_simd_mask(xmm::abi, src);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr mask<E>
    DPL_VECTORCALL to_simd_mask(
        assume_normalized_mask_t tag, vector<E> src) noexcept {
    return xmm::to_simd_mask(xmm::abi, tag, src);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
