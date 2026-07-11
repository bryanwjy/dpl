// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#if !__DPL_SHOULD_USE_BUILTIN(builtin_is_implicit_lifetime) && \
    !__DPL_SHOULD_USE_BUILTIN(is_implicit_lifetime)
#  include "dpl/std/type_traits/is_aggregate.h"
#  include "dpl/std/type_traits/is_array.h"
#  include "dpl/std/type_traits/is_copy_constructible.h"
#  include "dpl/std/type_traits/is_default_constructible.h"
#  include "dpl/std/type_traits/is_destructible.h"
#  include "dpl/std/type_traits/is_move_constructible.h"
#  include "dpl/std/type_traits/is_scalar.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(builtin_is_implicit_lifetime)
template <typename T>
inline constexpr bool is_implicit_lifetime_v =
    __builtin_is_implicit_lifetime(T);
template <typename T>
struct is_implicit_lifetime :
    bool_constant<__builtin_is_implicit_lifetime(T)> {};

#elif __DPL_SHOULD_USE_BUILTIN(is_implicit_lifetime)
template <typename T>
inline constexpr bool is_implicit_lifetime_v = __is_implicit_lifetime(T);
template <typename T>
struct is_implicit_lifetime : bool_constant<__is_implicit_lifetime(T)> {};

#else // if __DPL_SHOULD_USE_BUILTIN(is_implicit_lifetime)

template <typename T>
inline constexpr bool is_implicit_lifetime_v = is_scalar_v<T> ||
    is_array_v<T> || (is_aggregate_v<T> && is_trivially_destructible_v<T>) ||
    (is_trivially_destructible_v<T> &&
        (is_trivially_move_constructible_v<T> ||
            is_trivially_copy_constructible_v<T> ||
            is_trivially_default_constructible_v<T>));
template <typename T>
inline constexpr bool is_implicit_lifetime_v<T const> =
    is_implicit_lifetime_v<T>;
template <typename T>
inline constexpr bool is_implicit_lifetime_v<T volatile> =
    is_implicit_lifetime_v<T>;
template <typename T>
inline constexpr bool is_implicit_lifetime_v<T const volatile> =
    is_implicit_lifetime_v<T>;

template <typename T>
struct is_implicit_lifetime : bool_constant<is_implicit_lifetime_v<T>> {};

#endif // if __DPL_SHOULD_USE_BUILTIN(is_implicit_lifetime)

__DPL_DEFAULT_NAMESPACE_END
