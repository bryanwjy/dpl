// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/broadcastable_to.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/utility/forward.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
template <typename...>
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

template <simd_abi A, simd_element_for<A> E>
struct broadcast_t<A, E> {
public:
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_vector<E, A> operator()(E scalar) noexcept
    requires requires { broadcast<E>(internal::abi<A>, scalar); }
    {
        return broadcast<E>(internal::abi<A>, scalar);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        same_as<bool> auto scalar) noexcept
    requires requires { broadcast<E>(internal::abi<A>, scalar); }
    {
        return broadcast<E>(internal::abi<A>, scalar);
    }
};

template <simd_abi A, simd_element_for<A> E>
struct broadcast_t<E, A> : broadcast_t<A, E> {};

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

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr T operator()(same_as<bool> auto scalar) noexcept
    requires simd_mask<T> && regular_invocable<broadcast_t<A, E>, bool>
    {
        return broadcast_t<A, E>::operator()(scalar);
    }
};

template <simd_class T>
struct broadcast_t<T> {
private:
    using E DPL_NODEBUG = typename T::value_type;
    using base_type DPL_NODEBUG = broadcast_t<canonical_type_t<T>>;

public:
    template <typename Arg>
    requires regular_invocable<base_type, Arg>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr T operator()(Arg&& scalar) noexcept
    requires explicitly_convertible_to<canonical_type_t<T>, T>
    {
        // ADL cannot perform the lookup for
        // broadcast<T>(internal::abi<T>, scalar)
        return static_cast<T>(
            base_type::operator()( __DPL forward<Arg>(scalar)));
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
