// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/numbers/ext/common.h"
#include "dpl/core/numbers/floating_point_traits.h"

#if !DPL_MODULES
#  include "dpl/core/details/numbers.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept floating_point_like = details::numbers::floating_point_like<T>;

DPL_DEFAULT_NAMESPACE_END
