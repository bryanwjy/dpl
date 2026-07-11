// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_trivially_copyable.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

#ifndef DPL_BUILTIN_bit_cast
#  error "Unsupported compiler"
#endif

template <typename To, typename From>
requires (sizeof(To) == sizeof(From) && is_trivially_copyable_v<To> &&
    is_trivially_copyable_v<From>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr To bit_cast(From const& from) noexcept {
    return DPL_BUILTIN_bit_cast(To, from);
}

__DPL_DEFAULT_NAMESPACE_END
