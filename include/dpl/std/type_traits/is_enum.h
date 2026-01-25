// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_enum)
DPL_EXPORT template <typename T>
inline constexpr bool is_enum_v = __is_enum(T);
DPL_EXPORT template <typename T>
struct is_enum : bool_constant<__is_enum(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_enum)
DPL_EXPORT template <typename T>
inline constexpr bool is_enum_v = unsupported_trait_v<T>;
DPL_EXPORT template <typename T>
struct is_enum : unsupported_trait<T> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_enum)

DPL_DEFAULT_NAMESPACE_END
