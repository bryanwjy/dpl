// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/details/concepts.h"
#  include "dpl/std/type_traits/is_same.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T, typename U>
concept same_as = details::concepts::same_as<T, U>;

DPL_DEFAULT_NAMESPACE_END
