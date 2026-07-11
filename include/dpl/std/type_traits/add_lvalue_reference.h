// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(add_lvalue_reference)
template <typename T>
using add_lvalue_reference_t = __add_lvalue_reference(T);
template <typename T>
struct add_lvalue_reference {
    using type DPL_NODEBUG = __add_lvalue_reference(T);
};
#else  // if __DPL_SHOULD_USE_BUILTIN(add_lvalue_reference)

namespace details::type_traits {
class add_lvalue_reference {
    add_lvalue_reference() = delete;
    ~add_lvalue_reference() = delete;
    template <typename T>
    friend struct __DPL add_lvalue_reference;

    template <typename T>
    __DPL_HIDE_FROM_ABI static T&& make_result(int) noexcept;
    template <typename T>
    __DPL_HIDE_FROM_ABI static T make_result(float) noexcept;
};
} // namespace details::type_traits

template <typename T>
struct add_lvalue_reference {
    using type DPL_NODEBUG =
        decltype(details::type_traits::add_lvalue_reference::make_result<T>(0));
};

template <typename T>
using add_lvalue_reference_t = typename add_lvalue_reference<T>::type;
#endif // if __DPL_SHOULD_USE_BUILTIN(add_lvalue_reference)

__DPL_DEFAULT_NAMESPACE_END
