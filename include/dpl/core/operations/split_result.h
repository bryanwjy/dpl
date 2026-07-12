// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/std/concepts/derived_from.h"
#  include "dpl/std/type_traits/is_aggregate.h"
#  include "dpl/std/type_traits/is_constructible.h"
#  include "dpl/std/type_traits/is_explicitly_constructible.h"
#  include "dpl/std/type_traits/structured_bindings.h"
#  include "dpl/std/type_traits/tuple_like.h"
#  include "dpl/std/utility/forward.h"
#  include "dpl/std/utility/forward_like.h"
#  include "dpl/std/utility/move.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

template <simd_type T, size_t N>
struct split_result;

namespace internal {
template <typename U, typename T, typename S = make_index_sequence<T::size>>
inline constexpr bool is_constructible_from = false;

template <typename U, typename T, typename S = make_index_sequence<T::size>>
inline constexpr bool is_explicit_constructible_from = false;

template <size_t I, typename T>
using choose_type_t DPL_NODEBUG = T;

template <tuple_like U, size_t N, simd_type T, size_t... Is>
inline constexpr bool
    is_constructible_from<U, split_result<T, N>, index_sequence<Is...>> =
        is_constructible_v<U, choose_type_t<Is, T>...>;

template <tuple_like U, size_t N, simd_type T, size_t... Is>
inline constexpr bool is_explicit_constructible_from<U, split_result<T, N>,
    index_sequence<Is...>> =
    is_explicitly_constructible_v<U, choose_type_t<Is, T>...>;
} // namespace internal

template <simd_type T, size_t N>
struct split_result {
    static constexpr size_constant<N> size{};

    template <derived_from<split_result> S>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    constexpr decltype(auto) operator[](this S&& self, size_t idx) noexcept {
        return __DPL forward_like<S>(self.data_[idx]);
    }

    template <derived_from<split_result> S, tuple_like U>
    requires internal::is_constructible_from<U, split_result>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    explicit(internal::is_explicit_constructible_from<U, split_result>)
    operator U(this S&& self) noexcept {
        constexpr make_index_sequence<N> seq{};
        return [&]<size_t... Is>(index_sequence<Is...>) {
            if constexpr (is_aggregate_v<U>) {
                return U{__DPL forward_like<S>(self.data[Is])...};
            } else {
                return U(__DPL forward_like<S>(self.data[Is])...);
            }
        }(seq);
    }

    T data[N];
};

template <simd_type... Ts>
requires (... && convertible_to<Ts, common_type_t<Ts...>>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto make_split_result(Ts&&... data) noexcept {
    return split_result<common_type_t<Ts...>, sizeof...(Ts)>{
        static_cast<common_type_t<Ts...>>(__DPL forward<Ts>(data))...};
}

template <size_t I, size_t N, simd_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr decltype(auto) get(
    split_result<T, N> const&& data DPL_LIFETIMEBOUND) noexcept {
    return __DPL move(data[I]);
}

template <size_t I, size_t N, simd_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr decltype(auto) get(
    split_result<T, N> const& data DPL_LIFETIMEBOUND) noexcept {
    return data[I];
}

template <size_t I, size_t N, simd_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr decltype(auto) get(
    split_result<T, N>&& data DPL_LIFETIMEBOUND) noexcept {
    return __DPL move(data[I]);
}

template <size_t I, size_t N, simd_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr decltype(auto) get(
    split_result<T, N>& data DPL_LIFETIMEBOUND) noexcept {
    return data[I];
}

} // namespace datapar

template <datapar::simd_type T, size_t N>
struct tuple_size<datapar::split_result<T, N>> :
    integral_constant<size_t, N> {};

template <size_t I, datapar::simd_type T, size_t N>
struct tuple_element<I, datapar::split_result<T, N>> {
    using type = T;
};

__DPL_DEFAULT_NAMESPACE_END
