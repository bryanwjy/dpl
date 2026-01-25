// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/destructible.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_constructible.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T, typename... Args>
concept constructible_from = destructible<T> && is_constructible_v<T, Args...>;

DPL_DEFAULT_NAMESPACE_END
