// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/basic/abi.h"
#  include "dpl/xmm/basic/broadcast.h"

#  if !DPL_MODULES
#    include "dpl/core/immediate/constants/zero.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E> undefined() noexcept {
    if consteval {
        return xmm::broadcast<E>(dx::zero);
    } else {
        if constexpr (is_same_v<native_vector_t<E>, __m128>) {
            return _mm_undefined_ps();
        } else if constexpr (is_same_v<native_vector_t<E>, __m128d>) {
            return _mm_undefined_pd();
        } else if constexpr (!is_same_v<native_vector_t<E>, __m128i>) {
            return __DPL bit_cast<native_vector_t<E>>(_mm_undefined_si128());
        } else {
            return _mm_undefined_si128();
        }
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr vector<E> undefined(abi_tag) noexcept {
    return xmm::undefined<E>();
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
