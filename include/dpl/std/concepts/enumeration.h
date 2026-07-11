// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_enum.h"
#  include "dpl/std/type_traits/is_scoped_enum.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T>
concept enumeration = is_enum_v<T>;

template <typename T>
concept scoped_enumeration = enumeration<T> && is_scoped_enum_v<T>;

template <typename T>
concept unscoped_enumeration = enumeration<T> && !scoped_enumeration<T>;

__DPL_DEFAULT_NAMESPACE_END
