// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/std/concepts/derived_from.h"
#  include "dpl/std/type_traits/is_aggregate.h"
#  include "dpl/std/type_traits/is_constructible.h"
#  include "dpl/std/type_traits/is_explicitly_constructible.h"
#  include "dpl/std/utility/forward.h"
#  include "dpl/std/utility/structured_bindings.h"
#  include "dpl/std/utility/tuple_like.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <simd_class T, size_t N>
struct split_result;

namespace internal {
template <typename U, typename T, typename S = make_index_sequence<T::size>>
inline constexpr bool is_constructible_from = false;

template <typename U, typename T, typename S = make_index_sequence<T::size>>
inline constexpr bool is_explicit_constructible_from = false;

template <size_t I, typename T>
using choose_type_t DPL_NODEBUG = T;

template <tuple_like U, size_t N, simd_class T, size_t... Is>
inline constexpr bool
    is_constructible_from<U, split_result<T, N>, index_sequence<Is...>> =
        is_constructible_v<U, choose_type_t<Is, T>...>;

template <tuple_like U, size_t N, simd_class T, size_t... Is>
inline constexpr bool is_explicit_constructible_from<U, split_result<T, N>,
    index_sequence<Is...>> =
    is_explicitly_constructible_v<U, choose_type_t<Is, T>...>;
} // namespace internal

DPL_EXPORT template <simd_class T, size_t N>
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
        return []<size_t... Is>(index_sequence<Is...>) {
            if constexpr (is_aggregate_v<U>) {
                return U{__DPL forward_like<S>(self.data[Is])...};
            } else {
                return U(__DPL forward_like<S>(self.data[Is])...);
            }
        }(seq);
    }

    T data[N];
};

template <simd_class... Ts>
requires (... && convertible_to<Ts, common_type_t<Ts...>>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto make_split_result(Ts&&... data) noexcept {
    return split_result<common_type_t<Ts...>, sizeof...(N)>{
        static_cast<common_type_t<Ts...>>(__DPL forward<Ts>(data))...};
}

DPL_EXPORT template <size_t I, size_t N, simd_class T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr decltype(auto)
    get(split_result<T, N> const&& data DPL_LIFETIMEBOUND) noexcept {
    return __DPL move(data[I]);
}

DPL_EXPORT template <size_t I, size_t N, simd_class T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr decltype(auto)
    get(split_result<T, N> const& data DPL_LIFETIMEBOUND) noexcept {
    return data[I];
}

DPL_EXPORT template <size_t I, size_t N, simd_class T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr decltype(auto)
    get(split_result<T, N>&& data DPL_LIFETIMEBOUND) noexcept {
    return __DPL move(data[I]);
}

DPL_EXPORT template <size_t I, size_t N, simd_class T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr decltype(auto)
    get(split_result<T, N>& data DPL_LIFETIMEBOUND) noexcept {
    return data[I];
}

} // namespace datapar

DPL_EXPORT template <size_t N, datapar::simd_class T>
struct tuple_size<datapar::split_result<T, N>> :
    integral_constant<size_t, N> {};

DPL_EXPORT template <size_t I, datapar::simd_class T>
struct tuple_element<datapar::split_result<T, N>> {
    using type = T;
};

DPL_DEFAULT_NAMESPACE_END
