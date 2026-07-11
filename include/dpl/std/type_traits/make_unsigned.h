// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/type_traits/constants.h"
#include "dpl/std/type_traits/is_enum.h"
#include "dpl/std/type_traits/is_integral.h"

__DPL_DEFAULT_NAMESPACE_BEGIN

#if __DPL_SHOULD_USE_BUILTIN(make_unsigned)
template <typename T>
using make_unsigned_t = __make_unsigned(T);
template <typename T>
struct make_unsigned {
    using type DPL_NODEBUG = __make_unsigned(T);
};
#else // if __DPL_SHOULD_USE_BUILTIN(make_unsigned)

namespace details::type_traits {
struct no_unsigned_type {};

template <typename T>
consteval auto find_unsigned_entry() noexcept {
    return no_unsigned_type{};
}

template <typename T>
requires is_integral_v<T> || is_enum_v<T>
consteval auto find_unsigned_entry() noexcept {
    if constexpr (sizeof(T) <= sizeof(unsigned char)) {
        return make_unsigned<unsigned char>{};
    } else if constexpr (sizeof(T) <= sizeof(unsigned short)) {
        return make_unsigned<unsigned short>{};
    } else if constexpr (sizeof(T) <= sizeof(unsigned int)) {
        return make_unsigned<unsigned int>{};
    } else if constexpr (sizeof(T) <= sizeof(unsigned long)) {
        return make_unsigned<unsigned long>{};
    } else if constexpr (sizeof(T) <= sizeof(unsigned long long)) {
        return make_unsigned<unsigned long long>{};
    }
#  if DPL_SUPPORTS_INT128
    else if constexpr (sizeof(T) <= sizeof(__uint128_t)) {
        return make_unsigned<__uint128_t>{};
    }
#  endif
    else {
        return no_unsigned_type{};
    }
}

template <typename T>
using make_unsigned DPL_NODEBUG =
    decltype(details::type_traits::find_unsigned_entry<T>());
} // namespace details::type_traits

template <typename T>
struct make_unsigned : details::type_traits::make_unsigned<T> {};

template <typename T>
using make_unsigned_t = typename make_unsigned<T>::type;

template <typename T>
struct make_unsigned<T const> : make_unsigned<T> {};
template <typename T>
struct make_unsigned<T volatile> : make_unsigned<T> {};
template <typename T>
struct make_unsigned<T const volatile> : make_unsigned<T> {};
template <>
struct make_unsigned<bool> {};
template <>
struct make_unsigned<unsigned char> {
    using type DPL_NODEBUG = unsigned char;
};
template <>
struct make_unsigned<signed char> {
    using type DPL_NODEBUG = unsigned char;
};
template <>
struct make_unsigned<unsigned short> {
    using type DPL_NODEBUG = unsigned short;
};
template <>
struct make_unsigned<signed short> {
    using type DPL_NODEBUG = unsigned short;
};
template <>
struct make_unsigned<unsigned int> {
    using type DPL_NODEBUG = unsigned int;
};
template <>
struct make_unsigned<signed int> {
    using type DPL_NODEBUG = unsigned int;
};
template <>
struct make_unsigned<unsigned long> {
    using type DPL_NODEBUG = unsigned long;
};
template <>
struct make_unsigned<signed long> {
    using type DPL_NODEBUG = unsigned long;
};

#  ifdef UTL_SUPPORTS_INT128
template <>
struct make_unsigned<__uint128_t> {
    using type DPL_NODEBUG = __uint128_t;
};
template <>
struct make_unsigned<__int128_t> {
    using type DPL_NODEBUG = __uint128_t;
};
#  endif

#endif // if __DPL_SHOULD_USE_BUILTIN(make_unsigned)

__DPL_DEFAULT_NAMESPACE_END
