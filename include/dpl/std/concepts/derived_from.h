// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_base_of.h"
#  include "dpl/std/type_traits/is_convertible.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename Derived, typename Base>
concept derived_from = is_base_of_v<Base, Derived> &&
    is_convertible_v<Derived const volatile*, Base const volatile*>;

__DPL_DEFAULT_NAMESPACE_END
