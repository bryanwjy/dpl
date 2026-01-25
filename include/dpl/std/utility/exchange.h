// Copyright 2025-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/forward.h"
#include "dpl/std/utility/move.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/assignable_from.h"
#  include "dpl/std/concepts/move_constructible.h"
#  include "dpl/std/type_traits/is_nothrow_assignable.h"
#  include "dpl/std/type_traits/is_nothrow_constructible.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

DPL_EXPORT template <move_constructible T, typename U = T>
requires assignable_from<T&, U>
__DPL_HIDE_FROM_ABI constexpr T exchange(T& obj, U&& new_value) noexcept(
    is_nothrow_move_constructible_v<T> && is_nothrow_assignable_v<T&, U>) {
    T previous(__DPL move(obj));
    obj = __DPL forward<U>(new_value);
    return previous;
}

DPL_DEFAULT_NAMESPACE_END
