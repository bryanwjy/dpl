// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/type_traits/underlying_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
namespace internal {
template <typename T, typename U>
concept underlying_type_of = same_as<underlying_type_t<U>, T>;
template <typename T>
concept naturally_arithmetic_enum = unscoped_enumeration<T> && requires(T val) {
    requires !requires { operator+(val, val); } && requires {
        { val + val } -> underlying_type_of<T>;
    };
    requires !requires { operator-(val, val); } && requires {
        { val - val } -> underlying_type_of<T>;
    };
    requires !requires { operator*(val, val); } && requires {
        { val * val } -> underlying_type_of<T>;
    };
};
} // namespace internal
DPL_EXPORT template <typename T>
concept arithmetic_type = !same_as<T, bool> &&
    (integral<T> || floating_point<T> ||
        internal::naturally_arithmetic_enum<T>);

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
