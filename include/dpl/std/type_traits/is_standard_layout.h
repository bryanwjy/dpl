// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_standard_layout)
DPL_EXPORT template <typename T>
inline constexpr bool is_standard_layout_v = __is_standard_layout(T);
DPL_EXPORT template <typename T>
struct is_standard_layout : bool_constant<__is_standard_layout(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_standard_layout)
DPL_EXPORT template <typename T>
inline constexpr bool is_standard_layout_v = unsupported_trait_v<T>;
DPL_EXPORT template <typename T>
struct is_standard_layout : unsupported_trait<T> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_volatile)

DPL_DEFAULT_NAMESPACE_END
