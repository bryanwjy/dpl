// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"
// IWYU pragma: private, include "dpl/std/utility/bitset.h"

#include "dpl/std/utility/bitset/bitset_traits.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/equality_comparable.h"
#  include "dpl/std/details/bitset.h"
#  include "dpl/std/type_traits/remove_cv.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T>
concept bitset_type = details::utility::bitset_type<T>;

DPL_EXPORT template <typename T>
concept integral_bitset_type =
    bitset_type<T> && integral<typename remove_cv_t<T>::underlying_type>;

DPL_EXPORT template <typename T>
concept bitset_constant_like =
    details::utility::bitset_constant_like<remove_cv_t<T>>;

DPL_DEFAULT_NAMESPACE_END
