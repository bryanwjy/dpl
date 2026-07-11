// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/convertible_to.h"
#include "dpl/std/concepts/same_as.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/common_reference.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T, typename U>
concept common_reference_with =
    same_as<common_reference_t<T, U>, common_reference_t<U, T>> &&
    convertible_to<T, common_reference_t<T, U>> &&
    convertible_to<U, common_reference_t<T, U>>;

__DPL_DEFAULT_NAMESPACE_END
