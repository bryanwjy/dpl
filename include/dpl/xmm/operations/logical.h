// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_1
#  include "dpl/xmm/operations/reinterpret.h"

#  if !DPL_MODULES
#    include "dpl/xmm/basic/abi.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {
template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline bool all_of(mask<E> src) noexcept {
    return _mm_test_all_ones(+xmm::reinterpret<int8>(src));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline bool all_of(abi_tag, mask<E> src) noexcept {
    return xmm::all_of<E>(src);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline bool none_of(mask<E> src) noexcept {
#  if DPL_SIMD_X86_AVX
    if constexpr (is_same_v<E, float>) {
        return _mm_testz_ps(+src, +src);
    } else if constexpr (is_same_v<E, double>) {
        return _mm_testz_pd(+src, +src);
    } else
#  endif // if DPL_SIMD_X86_AVX
        if constexpr (is_integral_v<E>) {
            return _mm_testz_si128(+src, +src);
        } else {
            return xmm::none_of(xmm::reinterpret<int8>(src));
        }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline bool none_of(abi_tag, mask<E> src) noexcept {
    return xmm::none_of<E>(src);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline bool some_of(mask<E> src) noexcept {
    if constexpr (is_integral_v<E>) {
        return _mm_test_mix_ones_zeros(+src, +xmm::broadcast<E>(dx::all_bits));
    } else {
        return xmm::some_of(xmm::reinterpret<int8>(src));
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline bool some_of(abi_tag, mask<E> src) noexcept {
    return xmm::some_of<E>(src);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline bool any_of(mask<E> src) noexcept {
    return !xmm::none_of(src);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline bool any_of(abi_tag, mask<E> src) noexcept {
    return xmm::any_of<E>(src);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
