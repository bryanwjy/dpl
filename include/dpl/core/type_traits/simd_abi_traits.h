// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/simd_abi_type.h"
#include "dpl/core/type_traits/simd_element_representation.h"
#include "dpl/core/type_traits/simd_element_type.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/type_interface/enable_simd_abi.h"
#  include "dpl/core/type_traits/details/has_simd_members.h"
#  include "dpl/core/type_traits/details/simd_abi_size.h"
#  include "dpl/std/type_traits/conditional.h"
#  include "dpl/std/type_traits/is_function.h"
#  include "dpl/std/utility/ignore.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename, typename>
struct simd_abi_traits {};

DPL_EXPORT template <internal::has_simd_abi A>
requires (!internal::has_simd_element<A>)
struct simd_abi_traits<A> {

    template <typename E>
    using native_vector = typename A::template native_vector<E>;
    template <typename E>
    using native_mask = typename A::template native_mask<E>;

    using type = A;
    static constexpr A value = A{};
    consteval operator A(this simd_abi_traits) noexcept { return value; }
    consteval A operator()(this simd_abi_traits) noexcept { return value; }

    template <internal::has_representation_for<A> E = char>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static constexpr size_t size() noexcept {
        if constexpr (is_function_v<decltype(A::template size<E>)>) {
            return A::template size<E>();
        } else {
            static_assert(requires { typename size_constant<A::size>; });
            return A::size / sizeof(E);
        }
    }

    template <internal::has_representation_for<A> E = char>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
    static consteval size_t alignment() noexcept {
        if constexpr (requires { typename size_constant<A::alignment>; }) {
            return A::alignment;
        } else {
            return alignof(native_vector<E>);
        }
    }
};

DPL_EXPORT template <internal::has_simd_members T>
struct simd_abi_traits<T> :
    simd_abi_traits<simd_element_type_t<T>, simd_abi_type_t<T>> {};

DPL_EXPORT template <typename T, different_from<ignore_t> U>
requires internal::has_representation_for<U, T> ||
    internal::has_representation_for<T, U>
struct simd_abi_traits<T, U> : private internal::simd_abi_size<T, U> {
private:
    using A DPL_NODEBUG = conditional_t<enable_simd_abi<T>, T, U>;
    using E DPL_NODEBUG = conditional_t<enable_simd_abi<T>, U, T>;
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

    static constexpr size_t alignment = base_type::template alignment<E>();
    using size_base::size;
    static constexpr A value = A{};
    consteval operator A(this simd_abi_traits) noexcept { return value; }
    consteval A operator()(this simd_abi_traits) noexcept { return value; }
};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
