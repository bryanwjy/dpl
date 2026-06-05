// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/derived_from.h"
#  include "dpl/std/concepts/semiregular.h"
#  include "dpl/std/type_traits/is_empty.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
struct broadcastable_base {
    __DPL_HIDE_FROM_ABI constexpr ~broadcastable_base() noexcept = default;
};

DPL_EXPORT template <typename T, typename ValueType>
concept broadcastable_constant = derived_from<T, broadcastable_base<T>> &&
    is_empty_v<T> && semiregular<T> && convertible_to<T, ValueType> &&
    requires { typename integral_constant<ValueType, T{}>; };

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
