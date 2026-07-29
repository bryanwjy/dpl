// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/internal/abi.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/utility/forward.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename>
void initialize(...) noexcept = delete;

template <typename T, typename U = void>
struct initialize_t : public basic_operation_base<initialize_t<T, U>> {
    using operation_base<initialize_t<T, U>>::operator();
};

template <typename T, typename U>
struct operation_signature<initialize_t<T, U>> {
    static consteval void operator()(auto&&, auto&&...) noexcept
    requires ((same_as<void, U> && (simd_type<T> || simd_abi<T>)) ||
        (simd_abi<T> && simd_element_for<U, T>) ||
        (simd_abi<U> && simd_element_for<T, U>))
    {}
};

template <simd_vector T>
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
        make_canonical_vector_t<decay_t<common_type_t<Es...>>, A>;

public:
    template <typename... Args>
    requires requires { typename deduced_simd<Args...>; } &&
        cpo_invocable<initialize_t<deduced_simd<Args...>>, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr deduced_simd<Args...> operator()(Args&&... args) noexcept {
        return initialize_t<deduced_simd<Args...>>::operator()(
            __DPL forward<Args>(args)...);
    }
};

template <typename T, different_from<void> U>
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
    static constexpr make_canonical_vector_t<E, A> operator()(
        Args&&... args) noexcept
    requires requires {
        initialize<E>(internal::abi<A>, __DPL forward<Args>(args)...);
    }
    {
        return initialize<E>(internal::abi<A>, __DPL forward<Args>(args)...);
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
template <typename T, typename U = void>
inline constexpr internal::initialize_t<T, U> initialize{};
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
