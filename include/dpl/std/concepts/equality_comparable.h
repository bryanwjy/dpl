// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/details/concepts.h"
#  include "dpl/std/type_traits/common_reference.h"
#  include "dpl/std/type_traits/remove_reference.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept equality_comparable = details::concepts::equality_comparable<T>;

DPL_EXPORT template <typename T, typename U>
concept equality_comparable_with =
    details::concepts::equality_comparable_with<T, U>;

DPL_DEFAULT_NAMESPACE_END
