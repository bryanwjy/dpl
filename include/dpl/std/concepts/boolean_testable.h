// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/convertible_to.h"
#include "dpl/std/concepts/same_as.h"
#if !DPL_MODULES
#  include "dpl/std/details/concepts.h"
#  include "dpl/std/type_traits/is_object.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept boolean_testable =
    details::concepts::boolean_testable<T> && requires(T&& val) {
        { !static_cast<T&&>(val) } -> details::concepts::boolean_testable;
    };

DPL_DEFAULT_NAMESPACE_END
