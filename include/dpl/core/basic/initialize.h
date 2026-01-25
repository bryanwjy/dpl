// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/broadcastable_to.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/type_traits/basic_type.h"
#  include "dpl/core/type_traits/bit_type.h"
#  include "dpl/core/type_traits/element_count.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/bit/has_single_bit.h"
#  include "dpl/std/concepts/array_initializable.h"
#  include "dpl/std/concepts/convertible_to.h"
#  include "dpl/std/concepts/invocable.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/utility/forward.h"
#  include "dpl/std/utility/sequence.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename>
void initialize(...) noexcept = delete;

template <typename>
struct initialize_t {};

template <basic_simd_class T>
struct initialize_t<T> {
private:
    using E DPL_NODEBUG = typename T::value_type; // bool for masks
    using Array DPL_NODEBUG = E[element_count<T>];

public:
    template <core_convertible_to<E>... Args>
    requires array_initializable<Array, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr T operator()(Args&&... args) noexcept
    requires simd_type<T> && requires {
        { initialize<E>(internal::abi<T>, Array{}) } -> same_as<T>;
    }
    {
        Array buffer{__DPL forward<Args>(args)...};
        return initialize<E>(internal::abi<T>, buffer);
    }

    template <same_as<bool>... Bs>
    requires array_initializable<Array, Bs...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr T operator()(Bs... args) noexcept
    requires simd_mask_type<T> && requires {
        { initialize<E>(internal::abi<T>, Array{}) } -> same_as<T>;
    }
    {
        Array buffer{__DPL forward<Bs>(args)...};
        return initialize<E>(internal::abi<T>, buffer);
    }
};

template <simd_abi A>
struct initialize_t<A> {
private:
    template <typename... Es>
    requires requires {
        typename common_type_t<Es...>;
        requires simd_element<common_type_t<Es...>>;
    }
    using deduced_simd DPL_NODEBUG = basic_simd<common_type_t<Es...>, A>;

    template <same_as<bool>... Bs>
    requires (has_single_bit(sizeof...(Bs)) && A::size >= sizeof...(Bs))
    using deduced_mask DPL_NODEBUG =
        simd_mask<bit_type_t<(A::size / sizeof...(Bs)) * char_bit_v>, A>;

public:
    template <simd_element... Args>
    requires requires { typename deduced_simd<Args...>; } &&
        regular_invocable<initialize_t<deduced_simd<Args...>>, Args...>
    DPL_ATTRIBUTES(SYL_HIDE_FROM_ABI, NODISCARD)
    static constexpr deduced_simd<Args...> operator()(Args&&... args) noexcept {
        return initialize_t<deduced_simd<Args...>>::operator()(
            __DPL forward<Args>(args)...);
    }

    template <same_as<bool>... Bs>
    requires requires { typename deduced_mask<Bs...>; } &&
        regular_invocable<initialize_t<deduced_mask<Bs...>>, Bs...>
    DPL_ATTRIBUTES(SYL_HIDE_FROM_ABI, NODISCARD)
    static constexpr deduced_simd<Bs...> operator()(Bs... args) noexcept {
        return initialize_t<deduced_simd<Bs...>>::operator()(args...);
    }
};

template <simd_class T>
struct initialize_t<T> {
private:
    using E DPL_NODEBUG = typename T::value_type; // bool for masks
    using Array DPL_NODEBUG = E[element_count<T>];
    using base_type DPL_NODEBUG = initialize_t<basic_type_t<T>>;

public:
    template <typename... Args>
    requires regular_invocable<base_type, Args...>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr T operator()(Args&&... args) noexcept
    requires explicitly_convertible_to<basic_type_t<T>, T>
    {
        return static_cast<T>(
            base_type::operator()( __DPL forward<Args>(args)...));
    }
};
} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename T>
inline constexpr internal::initialize_t<T> initialize{};
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
