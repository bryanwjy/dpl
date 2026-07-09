// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/convertible_to.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept floating_point = is_floating_point_v<T>;

DPL_DEFAULT_NAMESPACE_END
