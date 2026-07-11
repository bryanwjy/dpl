// Copyright 2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/declval.h"
#include "dpl/std/type_traits/is_constructible.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T, typename... Ts>
inline constexpr bool is_explicitly_constructible_v = false;

template <typename T, typename Head, typename Mid, typename... Tail>
requires __DPL is_constructible_v<T, Head, Mid, Tail...>
inline constexpr bool is_explicitly_constructible_v<T, Head, Mid, Tail...> =
    !requires(void (*func)(T)) {
        func({__DPL declval<Head>(), __DPL declval<Mid>(),
            __DPL declval<Tail>()...});
    };

template <typename T, typename Head>
requires __DPL is_constructible_v<T, Head>
inline constexpr bool is_explicitly_constructible_v<T, Head> =
    !__DPL is_convertible_v<Head, T>;

template <typename T>
requires __DPL is_default_constructible_v<T>
inline constexpr bool is_explicitly_constructible_v<T> =
    !requires(void (*func)(T)) { func({}); };

template <typename T, typename... Args>
struct is_explicitly_constructible :
    bool_constant<is_explicitly_constructible_v<T, Args...>> {};

__DPL_DEFAULT_NAMESPACE_END
