// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_rvalue_reference)
DPL_EXPORT template <typename T>
inline constexpr bool is_rvalue_reference_v = __is_rvalue_reference(T);
DPL_EXPORT template <typename T>
struct is_rvalue_reference : bool_constant<__is_rvalue_reference(T)> {};
#else  // __DPL_SHOULD_USE_BUILTIN(is_rvalue_reference)
DPL_EXPORT template <typename T>
inline constexpr bool is_rvalue_reference_v = false;
DPL_EXPORT template <typename T>
inline constexpr bool is_rvalue_reference_v<T&&> = true;
DPL_EXPORT template <typename T>
struct is_rvalue_reference : bool_constant<is_rvalue_reference_v<T>> {};
#endif // __DPL_SHOULD_USE_BUILTIN(is_rvalue_reference)

DPL_DEFAULT_NAMESPACE_END
