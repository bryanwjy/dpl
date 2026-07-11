// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(remove_cvref)
template <typename T>
using remove_cvref_t = __remove_cvref(T);
template <typename T>
struct remove_cvref {
    using type DPL_NODEBUG = __remove_cvref(T);
};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

template <typename T>
struct remove_cvref {
    using type DPL_NODEBUG = T;
};
template <typename T>
struct remove_cvref<T const> {
    using type DPL_NODEBUG = T;
};
template <typename T>
struct remove_cvref<T volatile> {
    using type DPL_NODEBUG = T;
};
template <typename T>
struct remove_cvref<T const volatile> {
    using type DPL_NODEBUG = T;
};
template <typename T>
struct remove_cvref<T&> : remove_cvref<T> {};
template <typename T>
struct remove_cvref<T&&> : remove_cvref<T> {};

template <typename T>
using remove_cvref_t = typename remove_cvref<T>::type;
#endif // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

__DPL_DEFAULT_NAMESPACE_END
