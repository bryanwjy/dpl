// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_same.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::concepts {
template <typename T, typename U>
concept same_as = is_same_v<T, U>;
}

DPL_EXPORT template <typename T, typename U>
concept same_as =
    details::concepts::same_as<T, U> && details::concepts::same_as<U, T>;

DPL_DEFAULT_NAMESPACE_END
