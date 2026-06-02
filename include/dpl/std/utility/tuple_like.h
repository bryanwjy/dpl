// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#include "dpl/std/utility/move.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/type_traits/add_lvalue_reference.h"
#  include "dpl/std/type_traits/add_rvalue_reference.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#  include "dpl/std/utility/forward.h"
#  include "dpl/std/utility/sequence.h"
#  include "dpl/std/utility/structured_bindings.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace ranges::details {
template <size_t>
void get(...) noexcept = delete;

template <typename T>
concept has_size = requires {
    typename __DPL size_constant<std::tuple_size_v<remove_cvref_t<T>>>;
};

template <typename T, size_t I>
concept has_element = has_size<T> && requires {
    requires I < std::tuple_size_v<remove_cvref_t<T>>;
    typename std::tuple_element_t<I, remove_cvref_t<T>>;
};

template <typename T, size_t I>
concept has_member_get = has_size<T> && has_element<T, I> &&
    requires(T&& t) { __DPL forward<T>(t).template get<I>(); };

template <typename T, size_t I>
concept has_adl_get = has_size<T> && has_element<T, I> &&
    !has_member_get<T, I> && requires(T&& t) { get<I>(__DPL forward<T>(t)); };

template <size_t I>
struct get_element_t {
private:
    template <typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static consteval auto nothrow_member_get() noexcept {
        return false;
    }

    template <has_member_get<I> T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static consteval auto nothrow_member_get() noexcept {
        return noexcept(__DPL declval<T>().template get<I>());
    }

    template <typename T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static consteval auto nothrow_adl_get() noexcept {
        return false;
    }
    template <has_adl_get<I> T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static consteval auto nothrow_adl_get() noexcept {
        return noexcept(get<I>(__DPL declval<T>()));
    }

public:
    __DPL_HIDE_FROM_ABI constexpr explicit get_element_t() noexcept = default;

    template <has_adl_get<I> T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) operator()(
        T&& val DPL_LIFETIMEBOUND) noexcept(nothrow_adl_get<T>()) {
        return get<I>(__DPL forward<T>(val));
    }

    template <has_member_get<I> T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) operator()(
        T&& val DPL_LIFETIMEBOUND) noexcept(nothrow_member_get<T>()) {
        return __DPL forward<T>(val).template get<I>();
    }

    template <typename T, size_t N>
    requires (I < N) && (!has_adl_get<add_lvalue_reference_t<T[N]>, I>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr add_lvalue_reference_t<T> operator()(
        T (&array)[N]) noexcept {
        return array[I];
    }

    template <typename T, size_t N>
    requires (I < N) && (!has_adl_get<add_rvalue_reference_t<T[N]>, I>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr add_rvalue_reference_t<T> operator()(
        T (&&array)[N]) noexcept {
        return __DPL move(array[I]);
    }
};

} // namespace ranges::details

namespace ranges {
inline namespace cpo {
DPL_EXPORT template <size_t I>
inline constexpr details::get_element_t<I> get_element{};
}
} // namespace ranges

namespace details::tuple_like {
template <typename T>
struct sequence_for {};

template <__DPL ranges::details::has_size T>
struct sequence_for<T> {
    using type DPL_NODEBUG =
        __DPL make_index_sequence<std::tuple_size_v<remove_cvref_t<T>>>;
};
template <typename T, size_t N>
struct sequence_for<T[N]> {
    using type DPL_NODEBUG = __DPL make_index_sequence<N>;
};
template <typename T>
using sequence_for_t DPL_NODEBUG = typename sequence_for<T>::type;

template <typename T, typename S = sequence_for_t<T>>
inline constexpr bool is_tuple_like = false;
template <typename T, size_t... Is>
inline constexpr bool is_tuple_like<T, index_sequence<Is...>> =
    (... && regular_invocable<ranges::details::get_element_t<Is>, T>);

} // namespace details::tuple_like

DPL_EXPORT template <typename T>
concept tuple_like = requires {
    typename details::tuple_like::sequence_for_t<T>;
} && details::tuple_like::is_tuple_like<T>;

DPL_DEFAULT_NAMESPACE_END
