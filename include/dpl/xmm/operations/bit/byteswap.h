// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  if !DPL_MODULES
#    include "dpl/core/fwd.h"

#    include "dpl/xmm/basic/abi.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

namespace internal {

alignas(16) inline constexpr char byteswap_lut_epi64[]{
    7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8};

alignas(16) inline constexpr char byteswap_lut_epi32[]{
    3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12};

alignas(16) inline constexpr char byteswap_lut_epi16[]{
    1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};

} // namespace internal

template <simd_element E>
requires integral<E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, PURE, NODISCARD)
inline vector<E>
    DPL_VECTORCALL byteswap(vector<E> val) noexcept {
    if constexpr (sizeof(E) == 8) {
        auto const lut = _mm_load_si128(
            reinterpret_cast<__m128i const*>(internal::byteswap_lut_epi64));
        return _mm_shuffle_epi8(+val, lut);
    } else if constexpr (sizeof(E) == 4) {
        auto const lut = _mm_load_si128(
            reinterpret_cast<__m128i const*>(internal::byteswap_lut_epi32));
        return _mm_shuffle_epi8(+val, lut);
    } else if constexpr (sizeof(E) == 2) {
        auto const lut = _mm_load_si128(
            reinterpret_cast<__m128i const*>(internal::byteswap_lut_epi16));
        return _mm_shuffle_epi8(+val, lut);
    } else {
        static_assert(sizeof(E) == 1);
        return val;
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto DPL_VECTORCALL byteswap(abi_tag, vector<E> val) noexcept
requires requires { xmm::byteswap(val); }
{
    return xmm::byteswap(val);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
