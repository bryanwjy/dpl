// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/same_as.h"
#endif

#include "dpl/std/bit/popcount.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

template <unsigned_integral T>
requires (!same_as<bool, T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr bool has_single_bit(T val) noexcept {
    return __DPL popcount(val) == 1;
}

__DPL_DEFAULT_NAMESPACE_END
