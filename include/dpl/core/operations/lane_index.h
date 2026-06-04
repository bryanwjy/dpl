// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/basic/indices.h"
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/iota_sequence.h"
#  include "dpl/core/type_traits/rebind_simd.h"
#  include "dpl/core/type_traits/representation.h"
#  include "dpl/core/type_traits/simd_vector_type.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/utility/apply.h"
#  include "dpl/std/utility/ignore.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
template <typename, typename = __DPL ignore_t>
struct lane_index_t {};

/**
 * Lane indices must ALWAYS return canonical simd types
 */
template <typename>
void lane_index(...) noexcept = delete;

template <typename T, different_from<ignore_t> U>
requires (simd_abi<T> && simd_element_for<U, T>) ||
    (simd_abi<U> && simd_element_for<T, U>)
struct lane_index_t<T, U> {
    using A DPL_NODEBUG = conditional_t<simd_abi<T>, T, U>;
    using E DPL_NODEBUG = conditional_t<simd_abi<T>, U, T>;
    using I DPL_NODEBUG = signed_representation_t<E>;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_vector<I, A> fallback() noexcept {
        return __DPL apply(
            [](auto... idx) {
                return dx::initialize<I, A>(static_cast<I>(idx())...);
            },
            iota_sequence<E, A>);
    }

public:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_vector<I, A> operator()() noexcept
    requires scalable_abi<A> && requires { lane_index<E>(internal::abi<A>); }
    {
        return lane_index<E>(internal::abi<A>);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_vector<I, A> operator()() noexcept
    requires fixed_width_abi<A>
    {
        if consteval {
            return fallback();
        } else {
            if constexpr (requires { lane_index<E>(internal::abi<A>); }) {
                return lane_index<E>(internal::abi<A>);
            } else {
                return fallback();
            }
        }
    }
};

template <simd_class T>
struct lane_index_t<T> {
private:
    using E DPL_NODEBUG = simd_lane_type_t<T>;
    using A DPL_NODEBUG = typename T::abi_type;

public:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()() noexcept
    requires regular_invocable<lane_index_t<A, E>>
    {
        return lane_index_t<A, E>::operator()();
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename T, typename U = __DPL ignore_t>
inline constexpr internal::lane_index_t<T, U> lane_index{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
