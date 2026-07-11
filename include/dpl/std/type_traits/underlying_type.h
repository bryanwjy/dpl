// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(underlying_type)
template <typename T>
using underlying_type_t = __underlying_type(T);
template <typename T>
struct underlying_type {
    using type DPL_NODEBUG = __underlying_type(T);
};
#else  // if __DPL_SHOULD_USE_BUILTIN(underlying_type)
template <typename T>
using underlying_type_t = details::type_traits::unsupported_trait_t<T>;
template <typename T>
struct underlying_type : details::type_traits::unsupported_trait<T> {};
#endif // if __DPL_SHOULD_USE_BUILTIN(underlying_type)

__DPL_DEFAULT_NAMESPACE_END
