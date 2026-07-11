// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_floating_point.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T>
concept floating_point = is_floating_point_v<T>;

__DPL_DEFAULT_NAMESPACE_END
