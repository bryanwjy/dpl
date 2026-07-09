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
#else // if __DPL_SHOULD_USE_BUILTIN(add_pointer)

DPL_EXPORT namespace details::type_traits {
class add_pointer {
    add_pointer() = delete;
    ~add_pointer() = delete;
    template <typename T>
    friend struct __DPL add_pointer;

    template <typename T, typename = T&>
    __DPL_HIDE_FROM_ABI static remove_reference_t<T>* make_result(int) noexcept;
    template <typename T> // T is void
    __DPL_HIDE_FROM_ABI static auto make_result(float) noexcept
        -> decltype(static_cast<T* (*)(T const volatile*)>(0)(
            (void const volatile*)0));
    template <typename T>
    __DPL_HIDE_FROM_ABI static T make_result(...) noexcept;
};
} // namespace details::type_traits

DPL_EXPORT template <typename T>
struct add_pointer {
    using type DPL_NODEBUG =
        decltype(details::type_traits::add_pointer::make_result<T>(0));
};

DPL_EXPORT template <typename T>
using add_pointer_t = typename add_pointer<T>::type;

#endif // if __DPL_SHOULD_USE_BUILTIN(add_pointer)

DPL_DEFAULT_NAMESPACE_END
