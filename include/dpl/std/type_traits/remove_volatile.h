// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(remove_volatile)
template <typename T>
using remove_volatile_t = __remove_volatile(T);
template <typename T>
struct remove_volatile {
    using type DPL_NODEBUG = __remove_volatile(T);
};
#else // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

template <typename T>
struct remove_volatile {
    using type DPL_NODEBUG = T;
};

template <typename T>
struct remove_volatile<T volatile> {
    using type DPL_NODEBUG = T;
};

template <typename T>
using remove_volatile_t = typename remove_volatile<T>::type;

#endif // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

__DPL_DEFAULT_NAMESPACE_END
