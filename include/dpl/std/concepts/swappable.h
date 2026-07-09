// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/common_reference_with.h"

#if !DPL_MODULES
#  include "dpl/std/details/concepts.h"
#  include "dpl/std/type_traits/extent.h"
#  include "dpl/std/type_traits/is_nothrow_assignable.h"
#  include "dpl/std/type_traits/is_nothrow_constructible.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT namespace ranges {
inline namespace cpo {
inline constexpr __DPL details::concepts::swap_t swap{};
}
} // namespace ranges

DPL_EXPORT template <typename T, typename U>
concept swappable_with =
    common_reference_with<T, U> && requires(T && (*lhs)(), U && (*rhs)()) {
        ranges::swap(lhs(), lhs());
        ranges::swap(rhs(), rhs());
        ranges::swap(lhs(), rhs());
        ranges::swap(rhs(), lhs());
    };

DPL_EXPORT template <typename T>
concept swappable = requires(T& lhs, T& rhs) { ranges::swap(lhs, rhs); };

DPL_DEFAULT_NAMESPACE_END
