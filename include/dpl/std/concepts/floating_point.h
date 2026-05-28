// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/stddef.h"
#  include "dpl/std/type_traits/is_floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept floating_point = is_floating_point_v<T>;

namespace internal {
template <typename T>
inline constexpr bool is_brain_float = false;

#ifndef DPL_BUILTIN_bit_cast
#  error "Unsupported compiler"
#endif

template <floating_point T>
requires (sizeof(T) == sizeof(int16))
inline constexpr bool is_brain_float<T> = DPL_BUILTIN_bit_cast(
                                              T, static_cast<int16>(0x7f80)) >
    static_cast<T>(0);
} // namespace internal

DPL_EXPORT template <typename T>
concept brain_float = sizeof(T) == sizeof(int16) && internal::is_brain_float<T>;

DPL_DEFAULT_NAMESPACE_END
