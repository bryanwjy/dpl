// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(add_rvalue_reference)
DPL_EXPORT template <typename T>
using add_rvalue_reference_t = __add_rvalue_reference(T);
DPL_EXPORT template <typename T>
struct add_rvalue_reference {
    using type DPL_NODEBUG = __add_rvalue_reference(T);
};
#else  // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

namespace details {
template <typename T>
__DPL_HIDE_FROM_ABI T&& add_rvalue_reference(int) noexcept;
template <typename T>
__DPL_HIDE_FROM_ABI T add_rvalue_reference(float) noexcept;
} // namespace details

DPL_EXPORT template <typename T>
using add_rvalue_reference_t = decltype(details::add_rvalue_reference<T>(0));

DPL_EXPORT template <typename T>
struct add_rvalue_reference {
    using type DPL_NODEBUG = decltype(details::add_rvalue_reference<T>(0));
};
#endif // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

DPL_DEFAULT_NAMESPACE_END
