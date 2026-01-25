// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(remove_reference)
DPL_EXPORT template <typename T>
using remove_reference_t = __remove_reference(T);
DPL_EXPORT template <typename T>
struct remove_reference {
    using type DPL_NODEBUG = __remove_reference(T);
};
#else  // if __DPL_SHOULD_USE_BUILTIN(remove_reference)
DPL_EXPORT template <typename T>
struct remove_reference {
    using type DPL_NODEBUG = T;
};
DPL_EXPORT template <typename T>
struct remove_reference<T&&> {
    using type DPL_NODEBUG = T;
};
DPL_EXPORT template <typename T>
struct remove_reference<T&> {
    using type DPL_NODEBUG = T;
};
DPL_EXPORT template <typename T>
using remove_reference_t = typename remove_reference<T>::type;
#endif // if __DPL_SHOULD_USE_BUILTIN(remove_reference)

DPL_DEFAULT_NAMESPACE_END
