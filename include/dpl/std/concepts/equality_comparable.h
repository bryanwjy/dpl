// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/boolean_testable.h"
#include "dpl/std/concepts/common_reference_with.h"
#include "dpl/std/concepts/convertible_to.h"
#include "dpl/std/concepts/same_as.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/common_reference.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#  include "dpl/std/type_traits/remove_reference.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::concepts {
template <typename T, typename U>
concept weakly_equality_comparable_with =
    requires(remove_reference_t<T> const& t, remove_reference_t<U> const& u) {
        { t == u } -> boolean_testable;
        { t != u } -> boolean_testable;
        { u == t } -> boolean_testable;
        { u != t } -> boolean_testable;
    };

template <typename T, typename U,
    typename C = common_reference_t<T const&, U const&>>
concept comparison_common_with_impl =
    same_as<common_reference_t<T const&, U const&>,
        common_reference_t<U const&, T const&>> &&
    requires {
        requires convertible_to<T const&, C const&> ||
            convertible_to<T, C const&>;
        requires convertible_to<U const&, C const&> ||
            convertible_to<U, C const&>;
    };

template <typename T, typename U>
concept comparison_common_with =
    common_reference_with<remove_cvref_t<T> const&, remove_cvref_t<U> const&> &&
    comparison_common_with_impl<remove_cvref_t<T>, remove_cvref_t<U>>;
} // namespace details::concepts

template <typename T>
concept equality_comparable =
    details::concepts::weakly_equality_comparable_with<T, T>;

template <typename T, typename U>
concept equality_comparable_with = equality_comparable<T> &&
    equality_comparable<U> && details::concepts::comparison_common_with<T, U> &&
    equality_comparable<common_reference_t<remove_reference_t<T> const&,
        remove_reference_t<U> const&>> &&
    details::concepts::weakly_equality_comparable_with<T, U>;

__DPL_DEFAULT_NAMESPACE_END
