// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(remove_all_extents)
DPL_EXPORT template <typename T>
using remove_all_extents_t = __remove_all_extents(T);
DPL_EXPORT template <typename T>
struct remove_all_extents {
    using type DPL_NODEBUG = __remove_all_extents(T);
};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

template <typename T>
struct remove_all_extents {
    using type DPL_NODEBUG = T;
};
template <typename T>
struct remove_all_extents<T[]> {
    using type DPL_NODEBUG = typename remove_all_extents<T>::type;
};
template <typename T, size_t N>
struct remove_all_extents<T[N]> {
    using type DPL_NODEBUG = typename remove_all_extents<T>::type;
};

template <typename T>
using remove_all_extents_t = typename remove_all_extents<T>::type;
#endif // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

DPL_DEFAULT_NAMESPACE_END
