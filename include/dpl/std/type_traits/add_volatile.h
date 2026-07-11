// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/is_same.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(add_volatile)
template <typename T>
using add_volatile_t = __add_volatile(T);
template <typename T>
struct add_volatile {
    using type DPL_NODEBUG = __add_volatile(T);
};
#else // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

template <typename T>
struct add_volatile {
    using type DPL_NODEBUG = T const;
};

template <typename T>
struct add_volatile<T const> {
    using type DPL_NODEBUG = T const;
};

template <typename T>
using add_volatile_t = typename add_volatile<T>::type;

#endif // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

__DPL_DEFAULT_NAMESPACE_END
