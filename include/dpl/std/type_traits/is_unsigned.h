// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#if !__DPL_SHOULD_USE_BUILTIN(is_unsigned)
#  include "dpl/std/type_traits/is_arithmetic.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_unsigned)
template <typename T>
inline constexpr bool is_unsigned_v = __is_unsigned(T);
template <typename T>
struct is_unsigned : bool_constant<__is_unsigned(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_unsigned)

template <typename T>
inline constexpr bool is_unsigned_v = false;
template <typename T>
requires is_arithmetic_v<T>
inline constexpr bool is_unsigned_v = (static_cast<T>(0) < static_cast<T>(-1));

template <typename T>
struct is_unsigned : bool_constant<is_unsigned_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_unsigned)

__DPL_DEFAULT_NAMESPACE_END
