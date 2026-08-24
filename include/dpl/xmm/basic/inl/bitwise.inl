// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  if !DPL_MODULES
#    include "dpl/xmm/basic/abi.h"
#  endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm::fwd {

DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD) inline __m128i and_si128(
    __m128i lhs, __m128i rhs) noexcept {
    return _mm_and_si128(lhs, rhs);
}

} // namespace datapar::xmm::fwd

__DPL_DEFAULT_NAMESPACE_END

#endif
