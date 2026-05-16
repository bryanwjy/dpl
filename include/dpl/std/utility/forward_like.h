// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/is_const.h"
#  include "dpl/std/type_traits/is_lvalue_reference.h"
#  include "dpl/std/type_traits/remove_reference.h"
#  include "dpl/std/utility/as_const.h"
#  include "dpl/std/utility/move.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <typename T, typename U>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto&& forward_like(U&& x DPL_LIFETIMEBOUND) noexcept {
    constexpr bool is_adding_const = is_const_v<remove_reference_t<T>>;
    if constexpr (is_lvalue_reference_v<T&&>) {
        if constexpr (is_adding_const)
            return __DPL as_const(x);
        else
            return static_cast<U&>(x);
    } else if constexpr (is_adding_const)
        return __DPL move(__DPL as_const(x));
    else
        return __DPL move(x);
}

DPL_DEFAULT_NAMESPACE_END
