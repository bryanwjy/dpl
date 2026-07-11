// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/stddef/types.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::numbers {

// Template to prevent common base between float16 and bfloat16
template <typename T, typename V>
struct alignas(uint16) storage16 {
    static_assert(sizeof(V) == sizeof(uint16) && alignof(V) == alignof(V));
    V value;
};

} // namespace details::numbers
__DPL_DEFAULT_NAMESPACE_END
