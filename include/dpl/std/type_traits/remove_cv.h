// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(remove_cv)
template <typename T>
using remove_cv_t = __remove_cv(T);
template <typename T>
struct remove_cv {
    using type DPL_NODEBUG = __remove_cv(T);
};
#else // if __DPL_SHOULD_USE_BUILTIN(remove_cv)

template <typename T>
struct remove_cv {
    using type DPL_NODEBUG = T;
};

template <typename T>
struct remove_cv<T const> {
    using type DPL_NODEBUG = T;
};
template <typename T>
struct remove_cv<T volatile> {
    using type DPL_NODEBUG = T;
};

template <typename T>
struct remove_cv<T const volatile> {
    using type DPL_NODEBUG = T;
};

template <typename T>
using remove_cv_t = typename remove_cv<T>::type;

#endif // if __DPL_SHOULD_USE_BUILTIN(remove_cv)

__DPL_DEFAULT_NAMESPACE_END
