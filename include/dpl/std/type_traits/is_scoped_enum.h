// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#if !__DPL_SHOULD_USE_BUILTIN(is_scoped_enum)
#  include "dpl/std/type_traits/underlying_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(is_scoped_enum)

DPL_EXPORT template <typename T>
inline constexpr bool is_scoped_enum_v = __is_scoped_enum(T);
DPL_EXPORT template <typename T>
struct is_scoped_enum : bool_constant<__is_scoped_enum(T)> {};

#else

DPL_EXPORT template <typename T>
inline constexpr bool is_scoped_enum_v = false;

DPL_EXPORT template <typename T>
requires requires {
    typename underlying_type_t<T>;
    requires requires(void (*func)(underlying_type_t<T>), T val) { func(val); };
};
inline constexpr bool is_scoped_enum_v = true;

DPL_EXPORT template <typename T>
struct is_scoped_enum : bool_constant<is_scoped_enum_v(T)> {};

#endif // if __DPL_SHOULD_USE_BUILTIN(is_scoped_enum)

DPL_DEFAULT_NAMESPACE_END
