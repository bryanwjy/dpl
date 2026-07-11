// Copyright 2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/std/concepts/common_reference_with.h"
#include "dpl/std/concepts/same_as.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/add_lvalue_reference.h"
#  include "dpl/std/type_traits/common_type.h"
#  include "dpl/std/type_traits/declval.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T, typename U>
concept common_with = same_as<common_type_t<T, U>, common_type_t<U, T>> &&
    requires {
        static_cast<common_type_t<T, U>>(__DPL declval<T>());
        static_cast<common_type_t<T, U>>(__DPL declval<U>());
    } &&
    common_reference_with<add_lvalue_reference_t<T const>,
        add_lvalue_reference_t<U const>> &&
    common_reference_with<add_lvalue_reference_t<common_type_t<T, U>>,
        common_reference_t<add_lvalue_reference_t<T const>,
            add_lvalue_reference_t<U const>>>;

__DPL_DEFAULT_NAMESPACE_END
