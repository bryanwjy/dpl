// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/type_traits/make_unsigned.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <integral T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr make_unsigned_t<T> to_unsigned(T val) noexcept {
    return static_cast<make_unsigned_t<T>>(val);
}

__DPL_DEFAULT_NAMESPACE_END
