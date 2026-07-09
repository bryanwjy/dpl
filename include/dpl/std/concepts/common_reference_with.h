// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/details/concepts.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T, typename U>
concept common_reference_with = details::concepts::common_reference_with<T, U>;

DPL_DEFAULT_NAMESPACE_END
