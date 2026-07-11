// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_nothrow_destructible.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T>
concept destructible = is_nothrow_destructible_v<T>;

__DPL_DEFAULT_NAMESPACE_END
