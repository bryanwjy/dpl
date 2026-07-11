// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/details/fwd.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#  include "dpl/core/type_traits/enable_simd_abi.h"
#  include "dpl/std/type_traits/conditional.h"
#  include "dpl/std/type_traits/is_function.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename E, typename A>
concept has_representation_for = requires {
    typename simd_element_representation<A, E>::type;
    typename A::template native_vector<E>;
    typename A::template native_mask<E>;
};

template <typename, typename>
struct simd_abi_size {};
template <typename T, typename U>
concept abi_with_functional_size =
    enable_simd_abi<T> && has_representation_for<U, T> &&
    is_function_v<decltype(T::template size<U>)>;

template <typename T, typename U>
concept abi_with_fixed_size =
    enable_simd_abi<T> && has_representation_for<U, T> &&
    requires { typename size_constant<T::size>; };

template <typename T, typename U>
requires abi_with_functional_size<T, U> || abi_with_functional_size<U, T>
struct simd_abi_size<T, U> {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr size_t size() noexcept {
        using A DPL_NODEBUG = conditional_t<enable_simd_abi<T>, T, U>;
        using E DPL_NODEBUG = conditional_t<enable_simd_abi<T>, U, T>;
        return simd_abi_traits<A>::template size<E>();
    }
};

template <typename T, typename U>
requires abi_with_fixed_size<T, U> || abi_with_fixed_size<U, T>
struct simd_abi_size<T, U> {
    static constexpr auto size = []() {
        if constexpr (enable_simd_abi<T>) {
            return size_constant<T::size / sizeof(U)>{};
        } else {
            return size_constant<U::size / sizeof(T)>{};
        }
    }();
};
} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
