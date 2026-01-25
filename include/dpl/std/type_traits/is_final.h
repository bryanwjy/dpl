// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_final)
DPL_EXPORT template <typename T>
inline constexpr bool is_final_v = __is_final(T);
DPL_EXPORT template <typename T>
struct is_final : bool_constant<__is_final(T)> {};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_final)
DPL_EXPORT template <typename T>
inline constexpr bool is_final_v = unsupported_trait_v<T>;
DPL_EXPORT template <typename T>
struct is_final : unsupported_trait<T> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_final)

DPL_DEFAULT_NAMESPACE_END
