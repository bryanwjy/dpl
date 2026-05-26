// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_lane_type.h"
#  include "dpl/std/type_traits/constants.h"
#  include "dpl/std/utility/ignore.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename, typename = __DPL ignore_t>
struct simd_abi_traits {};

DPL_EXPORT template <simd_abi T>
struct simd_abi_traits<T> {

    template <typename E>
    using native_vector =
        typename T::template native_vector<simd_element_representation_t<T, E>>;

    template <typename E>
    using native_mask =
        typename T::template native_mask<simd_element_representation_t<T, E>>;

    using type = T;
    static constexpr T value = T{};

    consteval operator T(this simd_abi_traits) noexcept { return value; }

    consteval T operator()(this simd_abi_traits) noexcept { return value; }

    template <typename E = char>
    requires requires { typename simd_element_representation_t<T, E>; }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr size_t size() noexcept {
        if constexpr (scalable_vector<T>) {
            return T::template size<E>();
        } else {
            return T::size / sizeof(E);
        }
    }

    template <typename E = char>
    requires requires { typename simd_element_representation_t<T, E>; }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static consteval size_t max_size() noexcept {
        if constexpr (requires {
                          typename integral_constant<size_t, T::max_size>;
                      }) {
            return T::max_size / sizeof(E);
        } else {
            return simd_abi_traits::size<E>();
        }
    }

    template <typename E = char>
    requires requires { typename simd_element_representation_t<T, E>; }
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static consteval size_t alignment() noexcept {
        if constexpr (requires {
                          typename integral_constant<size_t, T::alignment>;
                      }) {
            return T::alignment;
        } else {
            return alignof(native_vector<E>);
        }
    }
};

DPL_EXPORT template <simd_class T>
struct simd_abi_traits<T> :
    simd_abi_traits<simd_lane_type_t<T>, typename T::abi_type> {};

namespace internal {
template <typename, typename>
struct simd_abi_size {};

template <typename T, typename U>
requires (scalable_abi<T> && simd_element_for<U, T>) ||
    (scalable_abi<U> && simd_element_for<T, U>)
struct simd_abi_size<T, U> {
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr size_t size() noexcept {
        using A DPL_NODEBUG = conditional_t<simd_abi<T>, T, U>;
        using E DPL_NODEBUG = conditional_t<simd_abi<T>, U, T>;
        return simd_abi_traits<A>::template size<E>();
    }
};

template <typename T, typename U>
requires (fixed_width_abi<T> && simd_element_for<U, T>) ||
    (fixed_width_abi<U> && simd_element_for<T, U>)
struct simd_abi_size<T, U> {
    static constexpr size_constant<simd_abi_traits<conditional_t<simd_abi<T>, T,
        U>>::template size<conditional_t<simd_abi<T>, U, T>>()>
        size{};
};
} // namespace internal

DPL_EXPORT template <typename T, different_from<ignore_t> U>
requires (simd_abi<T> && simd_element_for<U, T>) ||
    (simd_abi<U> && simd_element_for<T, U>)
struct simd_abi_traits<T, U> : private internal::simd_abi_size<T, U> {
private:
    using A DPL_NODEBUG = conditional_t<simd_abi<T>, T, U>;
    using E DPL_NODEBUG = conditional_t<simd_abi<T>, U, T>;
    static_assert(basic_element<simd_element_representation_t<A, E>>);
    using base_type DPL_NODEBUG = simd_abi_traits<A>;
    using size_base DPL_NODEBUG = internal::simd_abi_size<T, U>;

public:
    using type = A;
    using element_type = E;
    using representation_type = simd_element_representation_t<A, E>;
    using native_vector = typename base_type::template native_vector<E>;
    using native_mask = typename base_type::template native_mask<E>;

    consteval operator simd_abi_traits<A>(this simd_abi_traits) noexcept {
        return {};
    }
    static constexpr size_t max_size = base_type::template max_size<E>();
    static constexpr size_t alignment = base_type::template alignment<E>();
    using size_base::size;
    static constexpr A value = A{};
    consteval operator A(this simd_abi_traits) noexcept { return value; }
    consteval A operator()(this simd_abi_traits) noexcept { return value; }
};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
