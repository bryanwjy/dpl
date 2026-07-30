// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_tuple.h"
#  include "dpl/core/type_traits/enable_simd_tuple.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_native_type.h"
#  include "dpl/std/concepts/tuple_like.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#  include "dpl/std/utility/as_const.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
template <size_t>
void get(...) noexcept = delete;
template <size_t>
void set(...) noexcept = delete;

template <typename T, size_t I>
concept canonical_element_accesible =
    enable_simd_tuple<T> && __DPL details::concepts::has_tuple_element<T, I> &&
    __DPL details::concepts::has_tuple_size<T> && (!tuple_like<T>) &&
    (I < std::tuple_size_v<T>) && canonical_vector<std::tuple_element_t<I, T>>;

template <typename T, size_t I>
concept canonical_get_element =
    canonical_element_accesible<remove_cvref_t<T>, I> && requires {
        {
            get<I>(internal::abi<T>, internal::declarg<T>())
        } -> same_as<decay_t<std::tuple_element_t<I, remove_cvref_t<T>>>>;
    };

template <typename L, size_t I>
concept canonical_set_element = canonical_element_accesible<L, I> &&
    requires(std::tuple_element_t<I, L> rhs) {
        {
            set<I>(internal::abi<L>, internal::declarg<L const&>(), rhs)
        } -> same_as<L>;
    };

template <size_t I>
struct get_element_t : basic_operation_base<set_element_t<I>> {
    template <tuple_like T>
    requires enable_simd_tuple<remove_cvref_t<T>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr decltype(auto) operator()(T&& arg) noexcept(
        noexcept(__DPL ranges::get_element<I>(arg))) {
        return __DPL ranges::get_element<I>(arg);
    }

    template <canonical_get_element<I> T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr decltype(auto) operator()(T&& arg) noexcept {
        return get<I>(internal::abi<T>, __DPL forward<T>(arg));
    }
};

template <size_t I>
struct set_element_t : basic_operation_base<set_element_t<I>> {
private:
    using base_get = __DPL details::concepts::get_element_t<I>;

public:
    template <tuple_like L, simd_vector R>
    requires enable_simd_tuple<L> &&
        assignable_from<cpo_result_t<base_get, L&>, R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr L& operator()(L& lhs, R&& rhs) noexcept(
        noexcept(__DPL ranges::get_element<I>(lhs)) &&
        is_nothrow_assignable_v<cpo_result_t<base_get, L&>, R>) {
        __DPL ranges::get_element<I>(lhs) = __DPL forward<R>(rhs);
        return lhs;
    }

    template <canonical_set_element<I> L, canonical_vector R>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    constexpr L& operator()(L& lhs, R rhs) noexcept {
        return lhs = set<I>(internal::abi<L>, __DPL as_const(lhs), rhs);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
template <size_t I>
inline constexpr internal::get_element_t<I> get_element;
template <size_t I>
inline constexpr internal::set_element_t<I> set_element;
} // namespace cpo
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
