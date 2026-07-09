// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/details/bit.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int popcount(T val) noexcept {
#if DPL_HAS_BUILTIN(__builtin_popcountg)
    return __builtin_popcountg(val);
#else
    return __DPL details::bit::popcount(val);
#endif
}

#undef __DPL_popcount
DPL_DEFAULT_NAMESPACE_END
