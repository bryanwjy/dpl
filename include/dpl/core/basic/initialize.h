// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/has_single_bit.h"
#  include "dpl/std/concepts/array_initializable.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/utility/bitset.h"
#  include "dpl/std/utility/forward.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename>
void initialize(...) noexcept = delete;

template <typename, typename = ignore_t>
struct initialize_t {};

template <typename T, different_from<ignore_t> U>
requires (simd_abi<T> && simd_element_for<U, T>) ||
    (simd_abi<U> && simd_element_for<T, U>)
struct initialize_t<T, U> {
    using E DPL_NODEBUG = conditional_t<simd_abi<T>, U, T>;
    using A DPL_NODEBUG = conditional_t<simd_abi<T>, T, U>;

public:
    template <core_convertible_to<E>... Args>
    requires fixed_width_abi<A> &&
        array_initializable<E[simd_abi_traits<E, A>::size], Args...> &&
        (... && !same_as<bool, Args>)
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
        static constexpr basic_vector<E, A> operator()(Args&&... args) noexcept
    requires requires {
        initialize<E>(internal::abi<A>, __DPL forward<Args>(args)...);
    }
    {
        return initialize<E>(internal::abi<A>, __DPL forward<Args>(args)...);
    }

    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_mask<E, A> operator()(
        bitset<simd_abi_traits<A, E>::size> data) noexcept
    requires fixed_width_abi<A> &&
        requires { initialize<E>(internal::abi<A>, data); }
    {
        return initialize<E>(internal::abi<A>, data);
    }

    template <core_convertible_to<E>... Args>
    requires scalable_abi<A> &&
        array_initializable<E[sizeof...(Args)], Args...> &&
        (... && !same_as<bool, Args>)
        DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
        static constexpr basic_vector<E, A> operator()(Args&&... args) noexcept
    requires requires {
        initialize<E>(internal::abi<A>, __DPL forward<Args>(args)...);
    }
    {
        return initialize<E>(internal::abi<A>, __DPL forward<Args>(args)...);
    }

    template <size_t W>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_mask<E, A> operator()(bitset<W> data) noexcept
    requires scalable_abi<A> &&
        requires { initialize<E>(internal::abi<A>, data); }
    {
        return initialize<E>(internal::abi<A>, data);
    }
};

template <canonical_class T>
requires fixed_width_class<T>
struct initialize_t<T, ignore_t> {
private:
    using A DPL_NODEBUG = typename T::abi_type;
    using E DPL_NODEBUG = simd_lane_type_t<T>;

public:
    template <typename... Args>
    requires regular_invocable<initialize_t<A, E>, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr T operator()(Args&&... args) noexcept
    requires simd_vector<T>
    {
        return initialize_t<A, E>::operator()(__DPL forward<Args>(args)...);
    }

    template <size_t W>
    requires regular_invocable<initialize_t<A, E>, bitset<W>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr T operator()(bitset<W> data) noexcept
    requires simd_mask<T>
    {
        return initialize_t<A, E>::operator()(data);
    }
};

template <simd_abi A>
struct initialize_t<A, ignore_t> {
private:
    template <typename... Es>
    requires requires {
        typename common_type_t<Es...>;
        requires simd_element_for<decay_t<common_type_t<Es...>>, A>;
    }
    using deduced_simd DPL_NODEBUG =
        basic_vector<decay_t<common_type_t<Es...>>, A>;

    template <size_t W>
    requires fixed_width_abi<A> && (__DPL has_single_bit(W))
    using deduced_mask DPL_NODEBUG =
        basic_mask<bit_type_t<(A::size * char_bit_v / W)>, A>;

public:
    template <typename... Args>
    requires requires { typename deduced_simd<Args...>; } &&
        regular_invocable<initialize_t<deduced_simd<Args...>>, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr deduced_simd<Args...> operator()(Args&&... args) noexcept {
        return initialize_t<deduced_simd<Args...>>::operator()(
            __DPL forward<Args>(args)...);
    }

    template <size_t W>
    requires requires { typename deduced_mask<W>; } &&
        regular_invocable<initialize_t<deduced_mask<W>>, bitset<W>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr deduced_mask<W> operator()(bitset<W> data) noexcept {
        return initialize_t<deduced_mask<W>>::operator()(data);
    }
};

template <extended_class T>
struct initialize_t<T, ignore_t> {
private:
    using base_type DPL_NODEBUG = initialize_t<canonical_type_t<T>>;

public:
    template <typename... Args>
    requires regular_invocable<base_type, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr T operator()(Args&&... args) noexcept
    requires constructible_from<T, Args...> ||
        explicitly_convertible_to<canonical_type_t<T>, T>
    {
        if constexpr (constructible_from<T, Args...>) {
            return T(__DPL forward<Args>(args)...);
        } else {
            return static_cast<T>(
                base_type::operator()( __DPL forward<Args>(args)...));
        }
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename T, typename... Ts>
inline constexpr internal::initialize_t<T, Ts...> initialize{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
