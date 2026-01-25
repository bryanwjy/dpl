// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#if !__DPL_SHOULD_USE_BUILTIN(is_signed)
#  include "dpl/std/type_traits/is_arithmetic.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_signed)
DPL_EXPORT template <typename T>
inline constexpr bool is_signed_v = __is_signed(T);
DPL_EXPORT template <typename T>
struct is_signed : bool_constant<__is_signed(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_signed)

DPL_EXPORT template <typename T>
inline constexpr bool is_signed_v = false;
DPL_EXPORT template <typename T>
requires is_arithmetic_v<T>
inline constexpr bool is_signed_v = (static_cast<T>(-1) < static_cast<T>(0));

DPL_EXPORT template <typename T>
struct is_signed : bool_constant<is_signed_v<T>> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_signed)

DPL_DEFAULT_NAMESPACE_END
