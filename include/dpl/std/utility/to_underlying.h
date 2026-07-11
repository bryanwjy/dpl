// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/type_traits/underlying_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <enumeration T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr underlying_type_t<T> to_underlying(T val) noexcept {
    return static_cast<underlying_type_t<T>>(val);
}

__DPL_DEFAULT_NAMESPACE_END
