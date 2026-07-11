// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_convertible.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename From, typename To>
concept explicitly_convertible_to =
    requires(From&& val) { static_cast<To>(static_cast<From&&>(val)); };

template <typename From, typename To>
concept convertible_to =
    is_convertible_v<From, To> && explicitly_convertible_to<From, To>;

template <typename From, typename To>
concept core_convertible_to =
    is_core_convertible_v<From, To> && explicitly_convertible_to<From, To>;

__DPL_DEFAULT_NAMESPACE_END
