// Copyright 2023-2025 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/stddef/types.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

template <typename T, T N>
struct integral_constant {
    static constexpr T value = N;
    using value_type DPL_NODEBUG = T;
    using type DPL_NODEBUG = integral_constant;
    __DPL_HIDE_FROM_ABI constexpr operator value_type(
        this integral_constant) noexcept {
        return N;
    }

    __DPL_HIDE_FROM_ABI constexpr value_type operator()(
        this integral_constant) noexcept {
        return N;
    }
};

template <size_t N>
using size_constant = integral_constant<size_t, N>;
template <size_t N>
using index_constant = integral_constant<size_t, N>;
using npos_type = size_constant<(size_t)-1>;

template <bool B>
using bool_constant = integral_constant<bool, B>;
using true_type = bool_constant<true>;
using false_type = bool_constant<false>;

template <typename T, T N>
inline constexpr T integral_constant_v = N;

namespace details::type_traits {
template <typename...>
inline constexpr bool always_false_v = false;

template <typename T, typename... Ts>
struct unsupported_trait {
private:
    static_assert(always_false_v<T, Ts...>, "Unsupported type trait");

public:
    using type DPL_NODEBUG = void;
    static constexpr bool value = false;
};

template <typename T, typename... Ts>
inline constexpr bool unsupported_trait_v = unsupported_trait<T, Ts...>::value;

template <typename T, typename... Ts>
using unsupported_trait_t DPL_NODEBUG =
    typename unsupported_trait<T, Ts...>::type;
} // namespace details::type_traits

__DPL_DEFAULT_NAMESPACE_END
