// Copyright 2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/forward.h"      // IWYU pragma: keep
#include "dpl/std/utility/forward_like.h" // IWYU pragma: keep

#if !DPL_MODULES
#  include "dpl/std/concepts/tuple_like.h"
#  include "dpl/std/details/apply.h"
#  include "dpl/std/type_traits/declval.h"
#  include "dpl/std/type_traits/is_invocable.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#  include "dpl/std/type_traits/sequence.h"
#  include "dpl/std/type_traits/structured_bindings.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

DPL_EXPORT template <typename F, tuple_like T>
requires details::apply::is_applicable_v<F, T>
__DPL_HIDE_FROM_ABI constexpr decltype(auto) apply(F&& func,
    T&& tuple) noexcept(details::apply::is_nothrow_applicable_v<F, T>) {
#if (DPL_HAS_CXX26_EXTENSIONS || DPL_CXX26) && \
    __cpp_structured_bindings >= 202411L
    auto&& [... vals] = __DPL forward<T>(tuple);
    return __DPL invoke(
        __DPL forward<F>(func), __DPL forward<decltype(vals)>(vals)...);
#else
    return [&]<size_t... Is>(__DPL index_sequence<Is...>) -> decltype(auto) {
        return std::invoke(__DPL forward<F>(func),
            __DPL forward_like<T>(ranges::get_element<Is>(tuple))...);
    }(details::tuple::sequence_for<remove_cvref_t<T>>);
#endif
}

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif

DPL_DEFAULT_NAMESPACE_END
