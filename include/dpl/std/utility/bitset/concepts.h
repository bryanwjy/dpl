// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/bitset/bitset_traits.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/remove_cv.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T>
concept bitset_type = details::utility::bitset_type<T>;

template <typename T>
concept integral_bitset_type =
    bitset_type<T> && integral<typename remove_cv_t<T>::underlying_type>;

template <typename T>
concept bitset_constant_like =
    details::utility::bitset_constant_like<remove_cv_t<T>>;

__DPL_DEFAULT_NAMESPACE_END
