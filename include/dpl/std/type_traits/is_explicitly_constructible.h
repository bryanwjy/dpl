// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/declval.h"
#include "dpl/std/type_traits/is_constructible.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace details::explicitly_constructible {
template <typename T>
__DPL_HIDE_FROM_ABI void implicit_conv(T) noexcept;

} // namespace details::explicitly_constructible

DPL_EXPORT template <typename T, typename... Ts>
inline constexpr bool is_explicitly_constructible_v = false;

DPL_EXPORT template <typename T, typename Head, typename Mid, typename... Tail>
requires __DPL is_constructible_v<T, Head, Mid, Tail...>
inline constexpr bool is_explicitly_constructible_v<T, Head, Mid, Tail...> =
    !requires {
        details::explicitly_constructible::implicit_conv<T>(
            {__DPL declval<Head>(), __DPL declval<Mid>(),
                __DPL declval<Tail>()...});
    };

DPL_EXPORT template <typename T, typename Head>
requires __DPL is_constructible_v<T, Head>
inline constexpr bool is_explicitly_constructible_v<T, Head> =
    !__DPL is_convertible_v<Head, T>;

DPL_EXPORT template <typename T>
requires __DPL is_default_constructible_v<T>
inline constexpr bool is_explicitly_constructible_v<T> =
    !requires { details::explicitly_constructible::implicit_conv<T>({}); };

DPL_EXPORT template <typename T, typename... Args>
struct is_explicitly_constructible :
    bool_constant<is_explicitly_constructible_v<T, Args...>> {};

DPL_DEFAULT_NAMESPACE_END
