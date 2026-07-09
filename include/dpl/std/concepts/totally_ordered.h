// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/equality_comparable.h"

#if !DPL_MODULES
#  include "dpl/std/details/concepts.h"
#  include "dpl/std/type_traits/common_reference.h"
#  include "dpl/std/type_traits/remove_reference.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept totally_ordered =
    equality_comparable<T> && details::concepts::partially_ordered_with<T, T>;

DPL_EXPORT template <typename T, typename U>
concept totally_ordered_with = totally_ordered<T> && totally_ordered<U> &&
    equality_comparable_with<T, U> &&
    totally_ordered<common_reference_t<remove_reference_t<T> const&,
        remove_reference_t<U> const&>> &&
    details::concepts::partially_ordered_with<T, U>;

DPL_DEFAULT_NAMESPACE_END
