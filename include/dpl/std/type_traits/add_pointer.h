// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/is_same.h"
#include "dpl/std/type_traits/remove_reference.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(add_pointer)
DPL_EXPORT template <typename T>
using add_pointer_t = __add_pointer(T);
DPL_EXPORT template <typename T>
struct add_pointer {
    using type DPL_NODEBUG = __add_pointer(T);
};
#else // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

namespace details {
template <typename T, typename = T&>
__DPL_HIDE_FROM_ABI remove_reference_t<T>* add_ptr(int) noexcept;
template <typename T> // T is void
__DPL_HIDE_FROM_ABI auto add_ptr(float) noexcept
    -> decltype(static_cast<T* (*)(T const volatile*)>(0)(
        (void const volatile*)0));
template <typename T>
__DPL_HIDE_FROM_ABI T add_ptr(...) noexcept;
} // namespace details

DPL_EXPORT template <typename T>
struct add_pointer {
    using type DPL_NODEBUG = decltype(details::add_ptr(0));
};

DPL_EXPORT template <typename T>
using add_pointer_t = decltype(details::add_ptr(0));

#endif // if __DPL_SHOULD_USE_BUILTIN(is_aggregate)

DPL_DEFAULT_NAMESPACE_END
