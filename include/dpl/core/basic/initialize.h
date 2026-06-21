// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/internal/abi.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/std/bit/bit_type.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/has_single_bit.h"
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/utility/bitset.h"
#  include "dpl/std/utility/forward.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename>
void initialize(...) noexcept = delete;

template <typename T, typename U = __DPL ignore_t>
struct initialize_t : private basic_operation_base<initialize_t<T, U>> {
    using operation_base<initialize_t<T, U>>::operator();
};

template <typename T, typename U>
struct operation_signature<initialize_t<T, U>> {
    static consteval void operator()(auto&&, auto&&...) noexcept
    requires ((same_as<ignore_t, U> && (simd_type<T> || simd_abi<T>)) ||
        (simd_abi<T> && simd_element_for<U, T>) ||
        (simd_abi<U> && simd_element_for<T, U>))
    {}
};

template <simd_type T>
struct canonical_impl<initialize_t<T>> :
    canonical_impl<initialize_t<simd_abi_type_t<T>, simd_element_type_t<T>>> {};

template <simd_abi A>
struct canonical_impl<initialize_t<A>> {
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
        cpo_invocable<initialize_t<deduced_simd<Args...>>, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr deduced_simd<Args...> operator()(Args&&... args) noexcept {
        return initialize_t<deduced_simd<Args...>>::operator()(
            __DPL forward<Args>(args)...);
    }

    template <size_t W>
    requires requires { typename deduced_mask<W>; } &&
        cpo_invocable<initialize_t<deduced_mask<W>>, bitset<W>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr deduced_mask<W> operator()(bitset<W> data) noexcept {
        return initialize_t<deduced_mask<W>>::operator()(data);
    }
};

template <typename T, different_from<ignore_t> U>
requires (simd_abi<T> && simd_element_for<U, T>) ||
    (simd_abi<U> && simd_element_for<T, U>)
struct canonical_impl<initialize_t<T, U>> {
private:
    using E DPL_NODEBUG = conditional_t<simd_abi<T>, U, T>;
    using A DPL_NODEBUG = conditional_t<simd_abi<T>, T, U>;

public:
    template <convertible_to<E>... Args>
    requires (... && !same_as<bool, Args>)
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

    template <size_t W>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr basic_mask<E, A> operator()(bitset<W> data) noexcept
    requires scalable_abi<A> &&
        requires { initialize<E>(internal::abi<A>, data); }
    {
        return initialize<E>(internal::abi<A>, data);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename T, typename U = ignore_t>
inline constexpr internal::initialize_t<T, U> initialize{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
