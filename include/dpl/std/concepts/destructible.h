// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_nothrow_destructible.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept destructible = is_nothrow_destructible_v<T>;

DPL_DEFAULT_NAMESPACE_END
