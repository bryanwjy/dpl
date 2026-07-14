// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: private, include "dpl/xmm/operations/bitwise.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/operations/reinterpret.h"

#  if !DPL_MODULES
#    include "dpl/core/fwd.h"

#    include "dpl/core/type_traits/common_size_type.h"
#    include "dpl/core/type_traits/representation.h"
#    include "dpl/xmm/basic/abi.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL bwandnot(vector<E> lhs, vector<E> rhs) noexcept {
    if constexpr (is_same_v<__m128i, native_vector_t<E>>) {
        return _mm_andnot_si128(+rhs, +lhs);
    } else if constexpr (is_same_v<__m128, native_vector_t<E>>) {
        return _mm_andnot_ps(+rhs, +lhs);
    } else if constexpr (is_same_v<__m128d, native_vector_t<E>>) {
        return _mm_andnot_pd(+rhs, +lhs);
    } else {
        using rep = unsigned_representation_t<E>;
        auto const result = _mm_andnot_si128(
            +xmm::reinterpret<rep>(rhs), +xmm::reinterpret<rep>(lhs));
        return xmm::reinterpret<E>(vector<rep>(result));
    }
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_size_type_t<L, R>>
    DPL_VECTORCALL bwandnot(mask<L> lhs, mask<R> rhs) noexcept {
    return +xmm::bwandnot(vector<L>(+lhs), vector<R>(+rhs));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> bwandnot(abi_tag, vector<E> lhs, vector<E> rhs) noexcept
requires requires { xmm::bwandnot(lhs, rhs); }
{
    return xmm::bwandnot(lhs, rhs);
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto bwandnot(abi_tag, mask<L> lhs, mask<R> rhs) noexcept
requires requires { xmm::bwandnot(lhs, rhs); }
{
    return xmm::bwandnot(lhs, rhs);
}

} // namespace datapar::xmm

__DPL_DEFAULT_NAMESPACE_END

#endif
