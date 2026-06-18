// Copyright 2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/forward.h"      // IWYU pragma: keep
#include "dpl/std/utility/forward_like.h" // IWYU pragma: keep
#include "dpl/std/utility/sequence.h"
#include "dpl/std/utility/structured_bindings.h"
#include "dpl/std/utility/tuple_like.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/declval.h"
#  include "dpl/std/type_traits/is_invocable.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::apply {
template <size_t I, typename T>
requires requires { ranges::get_element<I>(__DPL declval<T>()); }
using decl_element_t DPL_NODEBUG =
    decltype(ranges::get_element<I>(__DPL declval<T>()));

template <typename T>
struct safe_sequence {
    using type DPL_NODEBUG = void;
};

template <__DPL tuple_like T>
struct safe_sequence<T> {
    using type DPL_NODEBUG =
        make_index_sequence<std::tuple_size_v<__DPL remove_cvref_t<T>>>;
};

template <typename F, typename T, typename = typename safe_sequence<T>::type>
inline constexpr bool is_applicable_v = false;

template <typename F, __DPL tuple_like T, size_t... Is>
inline constexpr bool is_applicable_v<F, T, index_sequence<Is...>> =
    __DPL is_invocable_v<F, decl_element_t<Is, T>...>;

template <typename F, typename T, typename = typename safe_sequence<T>::type>
inline constexpr bool is_nothrow_applicable_v = false;

template <typename F, __DPL tuple_like T, size_t... Is>
requires is_applicable_v<F, T>
inline constexpr bool is_nothrow_applicable_v<F, T, index_sequence<Is...>> =
    __DPL is_nothrow_invocable_v<F, decl_element_t<Is, T>...> &&
    (... && noexcept(ranges::get_element<Is>(__DPL declval<T>())));

} // namespace details::apply

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_PUSH()
DPL_DISABLE_WARNING("-Wc++26-extensions")
#endif

template <typename F, tuple_like T>
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
    }(details::tuple::sequence_for<std::remove_cvref_t<T>>);
#endif
}

#if DPL_HAS_CXX26_EXTENSIONS
DPL_DISABLE_WARNING_POP()
#endif

DPL_DEFAULT_NAMESPACE_END
