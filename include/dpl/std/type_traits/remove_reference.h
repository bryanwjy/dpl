// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(remove_reference)
template <typename T>
using remove_reference_t = __remove_reference(T);
template <typename T>
struct remove_reference {
    using type DPL_NODEBUG = __remove_reference(T);
};
#else  // if __DPL_SHOULD_USE_BUILTIN(remove_reference)
template <typename T>
struct remove_reference {
    using type DPL_NODEBUG = T;
};
template <typename T>
struct remove_reference<T&&> {
    using type DPL_NODEBUG = T;
};
template <typename T>
struct remove_reference<T&> {
    using type DPL_NODEBUG = T;
};
template <typename T>
using remove_reference_t = typename remove_reference<T>::type;
#endif // if __DPL_SHOULD_USE_BUILTIN(remove_reference)

__DPL_DEFAULT_NAMESPACE_END
