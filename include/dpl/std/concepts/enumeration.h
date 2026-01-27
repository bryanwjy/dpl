// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_enum.h"
#  include "dpl/std/type_traits/is_scoped_enum.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
requires is_enum_v<T>
inline constexpr bool enable_flag_enumeration = false;

DPL_EXPORT template <typename T>
concept enumeration = is_enum_v<T>;

DPL_EXPORT template <typename T>
concept scoped_enumeration = enumeration<T> && is_scoped_enum_v<T>;

DPL_EXPORT template <typename T>
concept unscoped_enumeration = enumeration<T> && !scoped_enumeration<T>;

DPL_EXPORT template <typename T>
concept flag_enumeration = enumeration<T> && enable_flag_enumeration<T>;

DPL_DEFAULT_NAMESPACE_END
