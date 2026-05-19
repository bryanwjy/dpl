// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/broadcastable_base.h"
#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/broadcastable_to.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/std/concepts/integral_constant_like.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/utility/forward.h"
#  include "dpl/std/utility/ignore.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
template <typename, typename>
struct broadcast_t {};

template <typename>
void broadcast(...) noexcept = delete;

template <simd_abi A>
struct broadcast_t<A> {
    template <typename E>
    using simd DPL_NODEBUG = basic_vector<E, A>;

public:
    template <simd_element_for<A> E>
    requires regular_invocable<broadcast_t<simd<E>>, E>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr simd<E> operator()(E scalar) noexcept {
        return broadcast_t<simd<E>>::operator()(scalar);
    }
};

template <typename T, different_from<ignore_t> U>
requires (simd_abi<T> && simd_element_for<U, T>) ||
    (simd_abi<U> && simd_element_for<T, U>)
struct broadcast_t<T, U> {
    using A DPL_NODEBUG = conditional_t<simd_abi<T>, T, U>;
    using E DPL_NODEBUG = conditional_t<simd_abi<T>, U, T>;

public:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_vector<E, A> operator()(E scalar) noexcept
    requires requires { broadcast<E>(internal::abi<A>, scalar); }
    {
        return broadcast<E>(internal::abi<A>, scalar);
    }

    template <broadcastable_constant<E> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_vector<E, A> operator()(V scalar) noexcept {
        return broadcast<E>(internal::abi<A>, scalar);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        same_as<bool> auto scalar) noexcept
    requires requires { broadcast<E>(internal::abi<A>, scalar); }
    {
        return broadcast<E>(internal::abi<A>, scalar);
    }

    template <integral_constant_like V>
    requires same_as<bool, typename V::value_type>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_mask<E, A> operator()(V scalar) noexcept {
        if constexpr (requires { broadcast<E>(internal::abi<A>, scalar); }) {
            return broadcast<E>(internal::abi<A>, scalar);
        } else {
            return broadcast<E>(internal::abi<A>, V::value);
        }
    }
};

template <canonical_class T>
struct broadcast_t<T> {
private:
    using E DPL_NODEBUG = typename T::value_type;
    using A DPL_NODEBUG = typename T::abi_type;

public:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(E scalar) noexcept
    requires simd_vector<T> && regular_invocable<broadcast_t<A, E>, E>
    {
        return broadcast_t<A, E>::operator()(scalar);
    }

    template <broadcastable_constant<E> V>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_vector<E, A> operator()(V scalar) noexcept
    requires simd_vector<T> && regular_invocable<broadcast_t<A, E>, V>
    {
        return broadcast_t<A, E>::operator()(scalar);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(same_as<bool> auto scalar) noexcept
    requires simd_mask<T> && regular_invocable<broadcast_t<A, E>, bool>
    {
        return broadcast_t<A, E>::operator()(scalar);
    }
};

template <extended_class T>
struct broadcast_t<T> {
private:
    using E DPL_NODEBUG = typename T::value_type;
    using base_type DPL_NODEBUG = broadcast_t<canonical_type_t<T>>;

public:
    template <typename Arg>
    requires regular_invocable<base_type, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr T operator()(Arg&& arg) noexcept
    requires constructible_from<T, broadcast_t, Arg> ||
        explicitly_convertible_to<canonical_type_t<T>, T>
    {
        if constexpr (constructible_from<T, broadcast_t, Arg>) {

            constexpr broadcast_t tag{};
            return T(tag, __DPL forward<Arg>(arg));
        } else {
            return static_cast<T>(
                base_type::operator()( __DPL forward<Arg>(arg)));
        }
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename T, typename... Ts>
inline constexpr internal::broadcast_t<T, Ts...> broadcast{};
}
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
