// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/extract.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/immediate.h"
#    include "dpl/std/bit/bit_cast.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD) constexpr bitset_t<E>
    DPL_VECTORCALL to_bitset(mask<E> src) noexcept {
    using bitset_t = bitset<vector<E>::size()>;
    if consteval {
        return [&]<size_t... Is>(index_sequence<Is...>) {
            return bitset_t(xmm::extract(src, imm<Is>)...);
        }(make_index_sequence<vector<E>::size()>{});
    } else {
        if constexpr (same_as<float, representation_t<E>>) {
            return bitset_t(static_cast<typename bitset_t::underlying_type>(
                __DPL to_unsigned(_mm_movemask_ps(+src))));
        } else if constexpr (same_as<double, representation_t<E>>) {
            return bitset_t(static_cast<typename bitset_t::underlying_type>(
                __DPL to_unsigned(_mm_movemask_pd(+src))));
        } else if constexpr (is_same_v<native_vector_t<E>, __m128i>) {
            if constexpr (sizeof(E) == sizeof(float)) {
                return xmm::to_bitset(mask<float>(_mm_castsi128_ps(+src)));
            } else if constexpr (sizeof(E) == sizeof(double)) {
                return xmm::to_bitset(mask<double>(_mm_castsi128_pd(+src)));
            } else if constexpr (sizeof(E) == sizeof(__DPL int8)) {
                return bitset_t(static_cast<typename bitset_t::underlying_type>(
                    __DPL to_unsigned(_mm_movemask_epi8(+src))));
            } else {
                static_assert(sizeof(E) == sizeof(__DPL uint16));
                auto const packed = _mm_packs_epi16(+src, _mm_setzero_si128());
                return bitset_t(static_cast<typename bitset_t::underlying_type>(
                    __DPL to_unsigned(_mm_movemask_epi8(packed))));
            }
        } else {
            return xmm::to_bitset(mask<unsigned_representation_t<E>>(
                __DPL bit_cast<__m128i>(+src)));
        }
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) constexpr bitset_t<E> to_bitset(
    abi_tag, mask<E> src) noexcept {
    return xmm::to_bitset(src);
}
} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
