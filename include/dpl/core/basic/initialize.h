// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/broadcast.h"
#include "dpl/core/basic/initializers.h"
#include "dpl/core/basic/internal/abi.h"
#include "dpl/core/basic/lane_index.h"
#include "dpl/core/basic/undefined.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/broadcastable_to.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/dispatch/interface.h"
#  include "dpl/core/dispatch/operation/basic.h"
#  include "dpl/core/immediate/broadcastable_base.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
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

    template <same_as<broadcasting_t> Tag, typename E>
    requires cpo_invocable<broadcast_t<T, U>, E> &&
        simd_vector<cpo_result_t<broadcast_t<T, U>, E>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr cpo_result_t<broadcast_t<T, U>, E>
    operator()(Tag, E&& scalar) noexcept(
        noexcept(dx::broadcast<T, U>(__DPL forward<E>(scalar)))) {
        return dx::broadcast<T, U>(__DPL forward<E>(scalar));
    }

    template <same_as<unspecified_t> Tag>
    requires cpo_invocable<undefined_t<T, U>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr cpo_result_t<undefined_t<T, U>> operator()(Tag) noexcept {
        return dx::undefined<T, U>();
    }

    template <same_as<iota_t> Tag>
    requires cpo_invocable<lane_index_t<T, U>>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr cpo_result_t<lane_index_t<T, U>> operator()(Tag) noexcept {
        return dx::lane_index<T, U>();
    }
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
    using E DPL_NODEBUG = typename simd_abi_traits<T, U>::element_type;
    using A DPL_NODEBUG = typename simd_abi_traits<T, U>::type;
    using vector_type DPL_NODEBUG = make_canonical_vector_t<E, A>;

public:
    template <convertible_to<E>... Args>
    requires (... && different_from<bool, Args>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr vector_type operator()(Args&&... args) noexcept
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
