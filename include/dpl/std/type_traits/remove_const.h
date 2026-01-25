// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/is_same.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(remove_const)
DPL_EXPORT template <typename T>
using remove_const_t = __remove_const(T);
DPL_EXPORT template <typename T>
struct remove_const {
    using type DPL_NODEBUG = __remove_const(T);
};
#else // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

DPL_EXPORT template <typename T>
struct remove_const {
    using type DPL_NODEBUG = T;
};

DPL_EXPORT template <typename T>
struct remove_const<T const> {
    using type DPL_NODEBUG = T;
};

DPL_EXPORT template <typename T>
using remove_const_t = typename remove_const<T>::type;

#endif // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

DPL_DEFAULT_NAMESPACE_END
