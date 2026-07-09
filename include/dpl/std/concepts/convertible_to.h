// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/details/concepts.h"
#  include "dpl/std/type_traits/is_convertible.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename From, typename To>
concept explicitly_convertible_to =
    details::concepts::explicitly_convertible_to<From, To>;

DPL_EXPORT template <typename From, typename To>
concept convertible_to = details::concepts::convertible_to<From, To>;

DPL_EXPORT template <typename From, typename To>
concept core_convertible_to =
    is_core_convertible_v<From, To> && explicitly_convertible_to<From, To>;

DPL_DEFAULT_NAMESPACE_END
