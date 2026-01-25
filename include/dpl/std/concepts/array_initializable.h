// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/convertible_to.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/extent.h"
#  include "dpl/std/type_traits/is_array.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T, typename... Args>
concept array_initializable = is_array_v<T> && extent_v<T> == sizeof...(Args) &&
    (... && core_convertible_to<Args, T>) && requires(Args&&... args) {
        { T{static_cast<Args&&>(args)...} };
    };

DPL_DEFAULT_NAMESPACE_END
