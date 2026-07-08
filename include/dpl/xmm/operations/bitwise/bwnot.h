// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: private, include "dpl/xmm/operations/bitwise.h"
#if !DPL_ARCH_x86_64 || !DPL_SIMD_X86_SSE4_2
#  error "Unsupported platform"
#endif

#include "dpl/xmm/operations/bitwise/bwxor.h"
#include "dpl/xmm/operations/reinterpret.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/xmm/basic/abi.h"

#  include <immintrin.h>
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL bwnot(vector<E> val) noexcept {
    auto const all = vector<int>(_mm_set1_epi32(-1));
    return xmm::bwxor(val, xmm::reinterpret<E>(all));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<E>
    DPL_VECTORCALL bwnot(mask<E> val) noexcept {
    return +xmm::bwnot(vector<E>(+val));
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto bwnot(abi_tag, vector<E> val) noexcept
requires requires { xmm::bwnot(val); }
{
    return xmm::bwnot(val);
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline auto bwnot(abi_tag, mask<E> val) noexcept
requires requires { xmm::bwnot(val); }
{
    return xmm::bwnot(val);
}

} // namespace datapar::xmm

DPL_DEFAULT_NAMESPACE_END
