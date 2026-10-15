// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if DPL_SIMD_X86_SSE4_2

#  include "dpl/xmm/basic/abi.h"

#  if DPL_MODULES
__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::xmm::fwd {

DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) __m128 castsi128_ps(__m128i) noexcept;
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) __m128d castsi128_pd(__m128i) noexcept;
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) __m128i castps_si128(__m128) noexcept;
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) __m128i castpd_si128(__m128d) noexcept;
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) __m128i castph_si128(__m128h) noexcept;
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD) __m128h castsi128_ph(__m128i) noexcept;

} // namespace datapar::xmm::fwd

__DPL_DEFAULT_NAMESPACE_END
#  else
#    include "dpl/xmm/basic/inl/cast.inl"
#  endif

#endif
