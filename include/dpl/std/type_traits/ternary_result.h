// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

DPL_DEFAULT_NAMESPACE_BEGIN
DPL_EXPORT namespace details::type_traits {
template <typename T, typename U>
using ternary_result_t DPL_NODEBUG =
    decltype(false ? static_cast<T (*)()>(0)() : static_cast<U (*)()>(0)());

template <typename, typename>
struct ternary_result {};

template <typename T, typename U>
requires requires { typename ternary_result_t<T, U>; }
struct ternary_result<T, U> {
    using type DPL_NODEBUG = ternary_result_t<T, U>;
};
} // namespace details::type_traits
DPL_DEFAULT_NAMESPACE_END
