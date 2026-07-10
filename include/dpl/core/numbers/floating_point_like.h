// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/numbers/details/floating_point_like.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept floating_point_like = details::numbers::floating_point_like<T>;

DPL_DEFAULT_NAMESPACE_END
