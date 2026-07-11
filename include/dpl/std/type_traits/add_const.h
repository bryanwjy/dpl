// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/is_same.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(add_const)
template <typename T>
using add_const_t = __add_const(T);
template <typename T>
struct add_const {
    using type DPL_NODEBUG = __add_const(T);
};
#else // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

template <typename T>
struct add_const {
    using type DPL_NODEBUG = T const;
};

template <typename T>
struct add_const<T const> {
    using type DPL_NODEBUG = T const;
};

template <typename T>
using add_const_t = typename add_const<T>::type;

#endif // if __DPL_SHOULD_USE_BUILTIN(add_const)

__DPL_DEFAULT_NAMESPACE_END
