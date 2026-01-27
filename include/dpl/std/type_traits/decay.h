// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/declval.h" // IWYU pragma: keep

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(decay)
DPL_EXPORT template <typename T>
using decay_t = __decay(T);
DPL_EXPORT template <typename T>
struct decay {
    using type DPL_NODEBUG = __decay(T);
};
#else // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

DPL_EXPORT template <typename T>
using decay_t =
    decltype([](decltype(__DPL declval<T>()) val) { return val; });

DPL_EXPORT template <typename T>
struct decay {
    using type DPL_NODEBUG = decay_t<T>;
};

#endif // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

DPL_DEFAULT_NAMESPACE_END
