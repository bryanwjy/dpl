// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#include "dpl/std/utility/bitset/bitset_traits.h"
#include "dpl/std/utility/structured_bindings.h"

#if !DPL_MODULES
#  include "dpl/std/bit/countl.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

template <size_t W>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr int bit_width(bitset<W> const& val) noexcept {
    return static_cast<int>(W) - __DPL countl_zero(val);
}

DPL_DEFAULT_NAMESPACE_END
