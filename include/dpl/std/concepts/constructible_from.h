// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/destructible.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_constructible.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T, typename... Args>
concept constructible_from = destructible<T> && is_constructible_v<T, Args...>;

__DPL_DEFAULT_NAMESPACE_END
