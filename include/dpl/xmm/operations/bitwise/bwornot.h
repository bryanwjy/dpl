// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

// IWYU pragma: private, include "dpl/xmm/operations/bitwise.h"
#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/operations/bitwise/bwnot.h"
#  include "dpl/xmm/operations/bitwise/bwor.h"
#  include "dpl/xmm/operations/reinterpret.h"

#  if !DPL_MODULES
#    include "dpl/core/fwd.h"

#    include "dpl/core/type_traits/common_size_type.h"
#    include "dpl/xmm/basic/abi.h"

#    include <immintrin.h>
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm {

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline vector<E>
    DPL_VECTORCALL bwornot(vector<E> lhs, vector<E> rhs) noexcept {
    return xmm::bwor(lhs, xmm::bwnot(rhs));
}

template <simd_element L, simd_element R>
requires common_size_with<L, R>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
inline mask<common_size_type_t<L, R>>
    DPL_VECTORCALL bwornot(mask<L> lhs, mask<R> rhs) noexcept {
    if constexpr (same_as<L, R>) {
        return +xmm::bwornot(vector<L>(+lhs), vector<R>(+rhs));
    } else {
        using uint_t = unsigned_representation_t<L>;
        return xmm::bwornot(
            xmm::reinterpret<uint_t>(lhs), xmm::reinterpret<uint_t>(rhs));
    }
}

template <simd_element E>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
inline vector<E> bwornot(abi_tag, vector<E> lhs, vector<E> rhs) noexcept
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

__DPL_DEFAULT_NAMESPACE_END

#endif
