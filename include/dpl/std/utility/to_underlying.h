// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_enum.h"
#  include "dpl/std/type_traits/underlying_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
requires is_enum_v<T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr underlying_type_t<T> to_underlying(T val) noexcept {
    return static_cast<underlying_type_t<T>>(val);
}

DPL_DEFAULT_NAMESPACE_END
