// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/boolean_testable.h"
#include "dpl/std/concepts/common_reference_with.h"
#include "dpl/std/concepts/convertible_to.h"
#include "dpl/std/concepts/equality_comparable.h"
#include "dpl/std/concepts/same_as.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/common_reference.h"
#  include "dpl/std/type_traits/remove_reference.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::concepts {
template <typename T, typename U>
concept partially_ordered_with =
    requires(remove_reference_t<T> const& t, remove_reference_t<U> const& u) {
        { t < u } -> boolean_testable;
        { t > u } -> boolean_testable;
        { t <= u } -> boolean_testable;
        { t >= u } -> boolean_testable;
        { u < t } -> boolean_testable;
        { u > t } -> boolean_testable;
        { u <= t } -> boolean_testable;
        { u >= t } -> boolean_testable;
    };

} // namespace details::concepts

template <typename T>
concept totally_ordered =
    equality_comparable<T> && details::concepts::partially_ordered_with<T, T>;

template <typename T, typename U>
concept totally_ordered_with = totally_ordered<T> && totally_ordered<U> &&
    equality_comparable_with<T, U> &&
    totally_ordered<common_reference_t<remove_reference_t<T> const&,
        remove_reference_t<U> const&>> &&
    details::concepts::partially_ordered_with<T, U>;

__DPL_DEFAULT_NAMESPACE_END
