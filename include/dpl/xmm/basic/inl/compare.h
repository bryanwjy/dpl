// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/basic/abi.h"

#  if DPL_MODULES
__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm::fwd {

DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) __m128i cmpeq_epi32(__m128i, __m128i) noexcept;
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) __m128i cmpgt_epi16(__m128i, __m128i) noexcept;
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) __m128i cmpgt_epi32(__m128i, __m128i) noexcept;
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) __m128i cmpgt_epi64(__m128i, __m128i) noexcept;

} // namespace datapar::xmm::fwd

__DPL_DEFAULT_NAMESPACE_END
#  else
#    include "dpl/xmm/basic/inl/compare.inl"
#  endif
#endif
