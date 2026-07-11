// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/numbers/details/extended_floating_point.h"
#include "dpl/core/numbers/floating_point_traits.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/derived_from.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T>
concept floating_point_like =
    (floating_point<T> ||
        derived_from<remove_cv_t<T>,
            details::numbers::extended_floating_point<remove_cv_t<T>>>) &&
    requires { typename floating_point_traits<T>::type; };

__DPL_DEFAULT_NAMESPACE_END
