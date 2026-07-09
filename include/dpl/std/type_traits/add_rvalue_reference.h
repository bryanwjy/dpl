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
#else  // if __DPL_SHOULD_USE_BUILTIN(add_rvalue_reference)

DPL_EXPORT namespace details::type_traits {
class add_rvalue_reference {
    add_rvalue_reference() = delete;
    ~add_rvalue_reference() = delete;
    template <typename T>
    friend struct __DPL add_rvalue_reference;

    template <typename T>
    __DPL_HIDE_FROM_ABI static T&& make_result(int) noexcept;
    template <typename T>
    __DPL_HIDE_FROM_ABI static T make_result(float) noexcept;
};
} // namespace details::type_traits

DPL_EXPORT template <typename T>
struct add_rvalue_reference {
    using type DPL_NODEBUG =
        decltype(details::type_traits::add_rvalue_reference::make_result<T>(0));
};

DPL_EXPORT template <typename T>
using add_rvalue_reference_t = typename add_rvalue_reference<T>::type;
#endif // if __DPL_SHOULD_USE_BUILTIN(add_rvalue_reference)

DPL_DEFAULT_NAMESPACE_END
