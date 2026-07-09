// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#include "dpl/std/type_traits/is_enum.h"
#include "dpl/std/type_traits/is_integral.h"

DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(make_signed)
DPL_EXPORT template <typename T>
using make_signed_t = __make_signed(T);
DPL_EXPORT template <typename T>
struct make_signed {
    using type DPL_NODEBUG = __make_signed(T);
};
#else // if __DPL_SHOULD_USE_BUILTIN(make_signed)

DPL_EXPORT namespace details::type_traits {

struct no_signed_type {};

template <typename T>
consteval auto find_signed_entry() noexcept {
    return no_signed_type{};
}

template <typename T>
requires is_integral_v<T> || is_enum_v<T>
consteval auto find_signed_entry() noexcept {
    if constexpr (sizeof(T) <= sizeof(signed char)) {
        return make_signed<signed char>{};
    } else if constexpr (sizeof(T) <= sizeof(signed short)) {
        return make_signed<signed short>{};
    } else if constexpr (sizeof(T) <= sizeof(signed int)) {
        return make_signed<signed int>{};
    } else if constexpr (sizeof(T) <= sizeof(signed long)) {
        return make_signed<signed long>{};
    } else if constexpr (sizeof(T) <= sizeof(signed long long)) {
        return make_signed<signed long long>{};
    }
#  if DPL_SUPPORTS_INT128
    else if constexpr (sizeof(T) <= sizeof(__int128_t)) {
        return make_signed<__int128_t>{};
    }
#  endif
    else {
        return no_signed_type{};
    }
}

template <typename T>
using make_signed DPL_NODEBUG =
    decltype(details::type_traits::find_signed_entry<T>());
} // namespace details::type_traits

DPL_EXPORT template <typename T>
struct make_signed : details::type_traits::make_signed<T> {};

DPL_EXPORT template <typename T>
using make_signed_t = typename make_signed<T>::type;

DPL_EXPORT template <typename T>
struct make_signed<T const> {
    using type DPL_NODEBUG = typename make_signed<T>::type const;
};
DPL_EXPORT template <typename T>
struct make_signed<T volatile> {
    using type DPL_NODEBUG = typename make_signed<T>::type volatile;
};
DPL_EXPORT template <typename T>
struct make_signed<T const volatile> {
    using type DPL_NODEBUG = typename make_signed<T>::type const volatile;
};
DPL_EXPORT template <>
struct make_signed<bool> {};
DPL_EXPORT template <>
struct make_signed<unsigned char> {
    using type DPL_NODEBUG = signed char;
};
DPL_EXPORT template <>
struct make_signed<signed char> {
    using type DPL_NODEBUG = signed char;
};
DPL_EXPORT template <>
struct make_signed<unsigned short> {
    using type DPL_NODEBUG = signed short;
};
DPL_EXPORT template <>
struct make_signed<signed short> {
    using type DPL_NODEBUG = signed short;
};
DPL_EXPORT template <>
struct make_signed<unsigned int> {
    using type DPL_NODEBUG = signed int;
};
DPL_EXPORT template <>
struct make_signed<signed int> {
    using type DPL_NODEBUG = signed int;
};
DPL_EXPORT template <>
struct make_signed<unsigned long> {
    using type DPL_NODEBUG = signed long;
};
DPL_EXPORT template <>
struct make_signed<signed long> {
    using type DPL_NODEBUG = signed long;
};

#  ifdef UTL_SUPPORTS_INT128
DPL_EXPORT template <>
struct make_signed<__uint128_t> {
    using type DPL_NODEBUG = __int128_t;
};
DPL_EXPORT template <>
struct make_signed<__int128_t> {
    using type DPL_NODEBUG = __int128_t;
};
#  endif

#endif // if __DPL_SHOULD_USE_BUILTIN(make_signed)

DPL_DEFAULT_NAMESPACE_END
