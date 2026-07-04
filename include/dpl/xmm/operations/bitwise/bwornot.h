// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: private, include "dpl/xmm/operations/bitwise.h"
#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/bitwise/bwnot.h"
#include "dpl/xmm/operations/bitwise/bwor.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/type_traits/common_size_type.h"
#  include "dpl/xmm/basic/abi.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline simd<E>
    DPL_VECTORCALL bwornot(simd<E> lhs, simd<E> rhs) noexcept {
    return xmm::bwor(lhs, xmm::bwnot(rhs));
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_size_type_t<L, R>>
    DPL_VECTORCALL bwornot(mask<L> lhs, mask<R> rhs) noexcept {
    return +xmm::bwornot(simd<L>(+lhs), simd<R>(+rhs));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline simd<E> bwornot(abi_tag, simd<E> lhs, simd<E> rhs) noexcept
requires requires { xmm::bwornot(lhs, rhs); }
{
    return xmm::bwornot(lhs, rhs);
}

template <simd_element L, simd_element R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto bwornot(abi_tag, mask<L> lhs, mask<R> rhs) noexcept
requires requires { xmm::bwornot(lhs, rhs); }
{
    return xmm::bwornot(lhs, rhs);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
