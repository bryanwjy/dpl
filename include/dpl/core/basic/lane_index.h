// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/std/utility/apply.h"
#  include "dpl/std/utility/ignore.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
/**
 * Lane indices must ALWAYS return canonical simd types
 */
template <typename>
void lane_index(...) noexcept = delete;

template <typename T, typename U = __DPL ignore_t>
struct lane_index_t : private basic_operation_base<lane_index_t<T, U>> {
    using operation_base<lane_index_t<T, U>>::operator();
};

template <typename T, typename U>
struct operation_signature<lane_index_t<T, U>> {
    static consteval void operator()() noexcept
    requires ((same_as<ignore_t, U> && (simd_type<T> || simd_abi<T>)) ||
        (simd_abi<T> && simd_element_for<U, T>) ||
        (simd_abi<U> && simd_element_for<T, U>))
    {}
};

template <typename E, typename A>
concept unqualified_canonical_lane_index = requires {
    {
        lane_index<E>(internal::abi<A>)
    } -> same_as<basic_vector<signed_representation_t<E>, A>>;
};

template <typename T, different_from<ignore_t> U>
requires (simd_abi<T> && simd_element_for<U, T>) ||
    (simd_abi<U> && simd_element_for<T, U>)
struct canonical_impl<lane_index_t<T, U>> {
private:
    using A DPL_NODEBUG = conditional_t<simd_abi<T>, T, U>;
    using E DPL_NODEBUG = conditional_t<simd_abi<T>, U, T>;
    using I DPL_NODEBUG = signed_representation_t<E>;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_vector<I, A> operator()() noexcept
    requires unqualified_canonical_lane_index<E, A>
    {
        return lane_index<E>(internal::abi<A>);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_vector<I, A> operator()() noexcept
    requires fixed_width_abi<A> && (!unqualified_canonical_lane_index<E, A>)
    {
        return __DPL apply(
            [](auto... idx) {
                return dx::initialize<I, A>(static_cast<I>(idx())...);
            },
            iota_sequence<E, A>);
    }
};

template <simd_abi A>
struct fallback_impl<lane_index_t<A>> :
    fallback_impl<lane_index_t<A, signed char>> {};

template <simd_type T>
struct fallback_impl<lane_index_t<T>> :
    fallback_impl<lane_index_t<simd_abi_type_t<T>, simd_element_type_t<T>>> {};

template <simd_abi A>
struct canonical_impl<lane_index_t<A>> :
    canonical_impl<lane_index_t<A, signed char>> {};

template <simd_type T>
struct canonical_impl<lane_index_t<T>> :
    canonical_impl<lane_index_t<simd_abi_type_t<T>, simd_element_type_t<T>>> {};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename T, typename U = __DPL ignore_t>
inline constexpr internal::lane_index_t<T, U> lane_index{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
