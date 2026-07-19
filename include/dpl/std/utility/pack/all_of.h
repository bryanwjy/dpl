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
constexpr bool all_of(F&& func, T&& tuple, Ts&&... tail) {
#if (DPL_HAS_CXX26_EXTENSIONS || DPL_CXX26) && \
    __cpp_expansion_statements >= 202506L
    template for (auto&& arg :
        {__DPL forward<T>(tuple), __DPL forward<Ts>(tail)...}) {
        template for (auto&& element : __DPL forward<decltype(arg)>(arg)) {
            if (!__DPL invoke(
                    func, __DPL forward<decltype(element)>(element))) {
                return false;
            }
        }
    }

    return true;
#else
    auto const invoke_on = [&]<typename... Es>(Es&&... element) {
        return (... && func(__DPL forward<Es>(element)));
    };
    return (dpl::apply(invoke_on, __DPL forward<T>(tuple)) && ... &&
        dpl::apply(invoke_on, dpl::forward<Ts>(tail)));
#endif
}
} // namespace pack

__DPL_DEFAULT_NAMESPACE_END

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif
