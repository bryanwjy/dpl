// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/common_reference_with.h"
#include "dpl/std/concepts/same_as.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_lvalue_reference.h"
#  include "dpl/std/type_traits/remove_reference.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename L, typename R>
concept assignable_from = is_lvalue_reference_v<L> &&
    common_reference_with<remove_reference_t<L> const&,
        remove_reference_t<R> const&> &&
    requires(L lhs, R && (*rhs)()) {
        { lhs = rhs() } -> same_as<L>;
    };

__DPL_DEFAULT_NAMESPACE_END
