// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/internal/abi.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/broadcastable_to.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/core/immediate/broadcastable_base.h"
#  include "dpl/core/type_traits/canonical_type.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
template <typename>
void broadcast(...) noexcept = delete;

template <typename T, typename U>
struct broadcast_t : public basic_operation_base<broadcast_t<T, U>> {
    using operation_base<broadcast_t<T, U>>::operator();
};

template <simd_abi A>
struct operation_signature<broadcast_t<A>> {
    static consteval void operator()(simd_element_for<A> auto) noexcept {}
    static consteval void operator()(bool) noexcept {}
    template <bool V>
    static consteval void operator()(bool_constant<V>) noexcept {}
};

template <simd_abi A>
struct canonical_impl<broadcast_t<A>> {
    template <simd_element_for<A> E>
    requires cpo_invocable<broadcast_t<A, E>, E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr make_canonical_vector_t<E, A> operator()(
        E scalar) noexcept {
        return broadcast_t<A, E>::operator()(scalar);
    }
};

template <simd_type T>
struct operation_signature<broadcast_t<T>> :
    operation_signature<
        broadcast_t<simd_abi_type_t<T>, simd_element_type_t<T>>> {};

template <simd_type T>
struct canonical_impl<broadcast_t<T>> :
    canonical_impl<broadcast_t<simd_abi_type_t<T>, simd_element_type_t<T>>> {};

template <typename T, different_from<void> U>
requires (simd_abi<T> && simd_element_for<U, T>) ||
    (simd_abi<U> && simd_element_for<T, U>)
struct operation_signature<broadcast_t<T, U>> {
    using A DPL_NODEBUG = conditional_t<simd_abi<T>, T, U>;
    using E DPL_NODEBUG = conditional_t<simd_abi<T>, U, T>;
    static consteval void operator()(E) noexcept {}
    static consteval void operator()(same_as<bool> auto) noexcept {}
    static consteval void operator()(broadcastable_constant<E> auto) noexcept {}
    static consteval void operator()(integral_constant_like auto) noexcept {}
};

template <typename T, different_from<void> U>
requires (simd_abi<T> && simd_element_for<U, T>) ||
    (simd_abi<U> && simd_element_for<T, U>)
struct canonical_impl<broadcast_t<T, U>> {
    using A DPL_NODEBUG = conditional_t<simd_abi<T>, T, U>;
    using E DPL_NODEBUG = conditional_t<simd_abi<T>, U, T>;

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr make_canonical_vector_t<E, A> operator()(E scalar) noexcept
    requires requires { broadcast<E>(internal::abi<A>, scalar); }
    {
        return broadcast<E>(internal::abi<A>, scalar);
    }

    template <broadcastable_constant<E> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr make_canonical_vector_t<E, A> operator()(V scalar) noexcept
    requires requires { broadcast<E>(internal::abi<A>, scalar); }
    {
        return broadcast<E>(internal::abi<A>, scalar);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr make_canonical_mask_t<E, A> operator()(
        same_as<bool> auto scalar) noexcept
    requires requires { broadcast<E>(internal::abi<A>, scalar); }
    {
        return broadcast<E>(internal::abi<A>, scalar);
    }

    template <bool_constant_like V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr make_canonical_mask_t<E, A> operator()(V scalar) noexcept {
        if constexpr (requires {
                          {
                              broadcast<E>(internal::abi<A>, scalar)
                          } -> same_as<make_canonical_mask_t<E, A>>;
                      }) {
            return broadcast<E>(internal::abi<A>, scalar);
        } else {
            return operator()(V::value);
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
template <typename T, typename U = void>
inline constexpr internal::broadcast_t<T, U> broadcast{};
}
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
