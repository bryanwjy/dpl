// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/numbers/details/fwd.h"

#include "dpl/core/numbers/details/extended_floating_point.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/derived_from.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/type_traits/remove_cv.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT namespace details::numbers {

template <typename T>
concept floating_point_like =
    (floating_point<T> ||
        derived_from<remove_cv_t<T>,
            extended_floating_point<remove_cv_t<T>>>) &&
    requires { typename floating_point_traits<T>::type; };

} // namespace details::numbers
DPL_DEFAULT_NAMESPACE_END
