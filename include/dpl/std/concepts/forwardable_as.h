// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/derived_from.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/copy_cvref.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename From, typename To>
concept forwardable_as =
    derived_from<remove_cvref_t<To>, remove_cvref_t<From>> &&
    convertible_to<From&&, copy_cvref_t<From, To>>;

DPL_DEFAULT_NAMESPACE_END
