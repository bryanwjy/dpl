// Copyright 2023-2026 Bryan Wong

#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/add_lvalue_reference.h"
#  include "dpl/std/type_traits/add_rvalue_reference.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#  include "dpl/std/type_traits/remove_reference.h"
#  include "dpl/std/type_traits/sequence.h"
#  include "dpl/std/type_traits/structured_bindings.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace details::concepts {
template <size_t>
void get(...) noexcept = delete;

template <typename T>
concept has_tuple_size = requires {
    std::tuple_size<remove_cvref_t<T>>::value;
    typename __DPL size_constant<std::tuple_size_v<remove_cvref_t<T>>>;
};

template <typename T, size_t I>
concept has_tuple_element = has_tuple_size<T> && requires {
    requires I < std::tuple_size_v<remove_cvref_t<T>>;
    typename std::tuple_element_t<I, remove_cvref_t<T>>;
};

template <typename T, size_t I>
concept has_tuple_member_get = has_tuple_size<T> && has_tuple_element<T, I> &&
    requires(T&& t) { static_cast<T&&>(t).template get<I>(); };

template <typename T, size_t I>
concept has_tuple_adl_get = has_tuple_size<T> && has_tuple_element<T, I> &&
    !has_tuple_member_get<T, I> &&
    requires(T&& t) { get<I>(static_cast<T&&>(t)); };

template <size_t I>
struct get_element_t {
private:
    template <typename T>
    DPL_NODISCARD static consteval auto nothrow_member_get() noexcept {
        return false;
    }

    template <has_tuple_member_get<I> T>
    DPL_NODISCARD static consteval auto nothrow_member_get() noexcept {
        using ref_t = T&&;
        using func_t = ref_t (*)();
        return noexcept(static_cast<func_t>(0)().template get<I>());
    }

    template <typename T>
    DPL_NODISCARD static consteval auto nothrow_adl_get() noexcept {
        return false;
    }
    template <has_tuple_adl_get<I> T>
    DPL_NODISCARD static consteval auto nothrow_adl_get() noexcept {
        using ref_t = T&&;
        using func_t = ref_t (*)();
        return noexcept(get<I>(static_cast<func_t>(0)()));
    }

public:
    __DPL_HIDE_FROM_ABI constexpr explicit get_element_t() noexcept = default;

    template <has_tuple_adl_get<I> T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) operator()(
        T&& val DPL_LIFETIMEBOUND) noexcept(nothrow_adl_get<T>()) {
        return get<I>(static_cast<T&&>(val));
    }

    template <has_tuple_member_get<I> T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr decltype(auto) operator()(
        T&& val DPL_LIFETIMEBOUND) noexcept(nothrow_member_get<T>()) {
        return static_cast<T&&>(val).template get<I>();
    }

    template <typename T, size_t N>
    requires (I < N) && (!has_tuple_adl_get<add_lvalue_reference_t<T[N]>, I>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr add_lvalue_reference_t<T> operator()(
        T (&array)[N]) noexcept {
        return array[I];
    }

    template <typename T, size_t N>
    requires (I < N) && (!has_tuple_adl_get<add_rvalue_reference_t<T[N]>, I>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr add_rvalue_reference_t<T> operator()(
        T (&&array)[N]) noexcept {
        return static_cast<remove_reference_t<T>&&>(array[I]);
    }
};

} // namespace details::concepts

namespace ranges {
inline namespace cpo {
template <size_t I>
inline constexpr details::concepts::get_element_t<I> get_element{};
}
} // namespace ranges

namespace details::concepts {
template <typename T>
struct sequence_for {};

template <__DPL details::concepts::has_tuple_size T>
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

template <typename T, size_t I>
concept gettable_from = requires(details::concepts::get_element_t<I> op,
    T&& arg) { op(static_cast<T&&>(arg)); };

template <typename T, typename S = sequence_for_t<T>>
inline constexpr bool is_tuple_like = false;
template <typename T, size_t... Is>
inline constexpr bool is_tuple_like<T, index_sequence<Is...>> =
    (... && gettable_from<T, Is>);

} // namespace details::concepts

template <typename T>
concept tuple_like = requires {
    typename details::concepts::sequence_for_t<remove_cvref_t<T>>;
} && details::concepts::is_tuple_like<T>;

__DPL_DEFAULT_NAMESPACE_END
