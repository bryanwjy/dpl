// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_reference)
template <typename T>
inline constexpr bool is_reference_v = __is_reference(T);
template <typename T>
struct is_reference : bool_constant<__is_reference(T)> {};
#else  // __DPL_SHOULD_USE_BUILTIN(is_reference)
template <typename T>
inline constexpr bool is_reference_v = false;
template <typename T>
inline constexpr bool is_reference_v<T&> = true;
template <typename T>
inline constexpr bool is_reference_v<T&&> = true;
template <typename T>
struct is_reference : bool_constant<is_reference_v<T>> {};
#endif // __DPL_SHOULD_USE_BUILTIN(is_reference)

__DPL_DEFAULT_NAMESPACE_END
