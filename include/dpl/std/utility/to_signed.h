// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/type_traits/make_signed.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <integral T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr make_signed_t<T> to_signed(T val) noexcept {
    return static_cast<make_signed_t<T>>(val);
}

DPL_DEFAULT_NAMESPACE_END
