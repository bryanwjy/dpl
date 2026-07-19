// Copyright 2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/apply.h"
#include "dpl/std/utility/pack/any_of.h"

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace pack {
template <typename F, __DPL tuple_like T, __DPL tuple_like... Ts>
requires (details::utility::tuple_element_predicate<F, T> && ... &&
    details::utility::tuple_element_predicate<F, Ts>)
constexpr bool none_of(F&& func, T&& tuple, Ts&&... tail) {
    return !pack::any_of(__DPL forward<F>(func), __DPL forward<T>(tuple),
        __DPL forward<Ts>(tail)...);
}
} // namespace pack

__DPL_DEFAULT_NAMESPACE_END

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif
