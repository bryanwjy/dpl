// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/internal/array_for.h"

#if !DPL_MODULES
#  include "dpl/core/basic/initialize.h"
#  include "dpl/core/basic/internal/abi.h"
#  include "dpl/core/basic/internal/iota_sequence.h"
#  include "dpl/core/basic/load.h"
#  include "dpl/core/basic/store.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/extended.h"
#  include "dpl/core/dispatch/operation/primitive.h"
#  include "dpl/core/immediate/immediate.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/type_traits/is_const.h"
#  include "dpl/std/type_traits/is_nothrow_constructible.h"
#  include "dpl/std/type_traits/is_object.h"
#  include "dpl/std/type_traits/is_volatile.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename>
void reinterpret(...) noexcept = delete;

template <typename E>
struct reinterpret_t : private primitive_operation_base<reinterpret_t<E>> {
    using operation_base<reinterpret_t<E>>::operator();
};

// TODO: remove this specialization
template <simd_type T>
struct reinterpret_t<T> : private reinterpret_t<simd_element_type_t<T>> {
    static_assert(is_object_v<T> && !is_const_v<T> && !is_volatile_v<T>);
    template <simd_type U>
    requires cpo_invocable<reinterpret_t<simd_element_type_t<T>>, U>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(U&& val) noexcept(canonical_simd_type<U>) {
        static_assert(same_as<simd_abi_type_t<T>, simd_abi_type_t<U>>);
        using base DPL_NODEBUG = reinterpret_t<simd_element_type_t<T>>;
        return base::operator()(__DPL forward<U>(val));
    }
};

template <typename E>
struct operation_signature<reinterpret_t<E>> {
    static consteval void operator()(simd_type auto&&) noexcept {}
};

template <typename ToE>
struct fallback_impl<reinterpret_t<ToE>> {
    template <simd_type T>
    requires same_as<simd_element_type_t<T>, ToE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) noexcept(
        is_nothrow_constructible_v<remove_cvref_t<T>, T>) {
        return __DPL forward<T>(val);
    }

    template <fixed_width_abi A, simd_element_for<A> FromE>
    requires simd_element_for<ToE, A> &&
        (sizeof(basic_vector<FromE, A>) == sizeof(basic_vector<ToE, A>))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_vector<ToE, A> operator()(
        basic_vector<FromE, A> from) noexcept {
        using to_vector = typename simd_abi_traits<ToE, A>::native_vector;
        using from_vector = typename simd_abi_traits<FromE, A>::native_vector;
        if constexpr (same_as<to_vector, from_vector>) {
            return +from;
        } else if consteval {
            // Workaround MSVC's unions
            array_for<FromE, A> buffer;
            dx::store(from, buffer.data);
            return dx::load<A>(__DPL bit_cast<array_for<ToE, A>>(buffer).data);
        } else {
            return __DPL bit_cast<basic_vector<ToE, A>>(from);
        }
    }

    template <fixed_width_abi A, simd_element_for<A> FromE>
    requires simd_element_for<ToE, A> && common_size_with<FromE, ToE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<ToE, A> operator()(
        basic_mask<FromE, A> from) noexcept {
        using to_mask = typename simd_abi_traits<ToE, A>::native_mask;
        using from_mask = typename simd_abi_traits<FromE, A>::native_mask;
        constexpr auto width = typename simd_abi_traits<FromE, A>::size();
        if constexpr (same_as<to_mask, from_mask>) {
            return +from;
        } else if consteval {
            // Workaround MSVC's unions
            return [&]<size_t... Is>(index_sequence<Is...>) {
                return dx::initialize<ToE, A>(bitset<width>(from[imm<Is>]...));
            }(iota_sequence<FromE, A>);
        } else {
            return __DPL bit_cast<basic_mask<ToE, A>>(from);
        }
    }
};

template <typename ToE>
struct canonical_impl<reinterpret_t<ToE>> {
    template <simd_abi A, simd_element_for<A> FromE>
    requires (simd_element_for<ToE, A> &&
        (sizeof(basic_vector<FromE, A>) == sizeof(basic_vector<ToE, A>)))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr basic_vector<ToE, A> operator()(
        basic_vector<FromE, A> val) noexcept
    requires requires { reinterpret<ToE>(internal::abi<A>, val); }
    {
        return reinterpret<ToE>(internal::abi<A>, val);
    }

    template <fixed_width_abi A, simd_element_for<A> FromE>
    requires (simd_element_for<ToE, A> && common_size_with<FromE, ToE>)
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr basic_mask<ToE, A> operator()(
        basic_mask<FromE, A> val) noexcept
    requires requires { reinterpret<ToE>(internal::abi<A>, val); }
    {
        return reinterpret<ToE>(internal::abi<A>, val);
    }
};

template <typename T, typename From, typename E>
concept extended_reinterpreted_result =
    common_simd_type_with<T, From> && same_as<simd_element_type_t<T>, E>;

template <typename From, typename To>
concept unqualified_extended_reinterpret = requires(From from) {
    { reinterpret<To>(from) } -> extended_reinterpreted_result<From, To>;
};

template <typename ToE>
struct extended_impl<reinterpret_t<ToE>> {
    template <extended_simd_type T>
    requires simd_element_for<simd_abi_type_t<T>, ToE> &&
        unqualified_extended_reinterpret<T, ToE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr auto operator()(T&& val) {
        return reinterpret<ToE>(__DPL forward<T>(val));
    }
};

} // namespace datapar::internal

namespace datapar {
inline namespace cpo {
DPL_EXPORT template <typename To>
inline constexpr internal::reinterpret_t<To> reinterpret{};
} // namespace cpo
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
