// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(remove_extent)
template <typename T>
using remove_extent_t = __remove_extent(T);
template <typename T>
struct remove_extent {
    using type DPL_NODEBUG = __remove_extent(T);
};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

template <typename T>
struct remove_extent {
    using type DPL_NODEBUG = T;
};
template <typename T>
struct remove_extent<T[]> {
    using type DPL_NODEBUG = T;
};
template <typename T, size_t N>
struct remove_extent<T[N]> {
    using type DPL_NODEBUG = T;
};
template <typename T>
using remove_extent_t = typename remove_extent<T>::type;
#endif // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

__DPL_DEFAULT_NAMESPACE_END
