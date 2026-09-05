// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/operations/internal/array_for.h"

#if !DPL_MODULES
#  include "dpl/core/basic/from_bitset.h"
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

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename>
void reinterpret(...) noexcept = delete;

template <typename E>
struct reinterpret_t : public primitive_operation_base<reinterpret_t<E>> {
    using operation_base<reinterpret_t<E>>::operator();
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
    static constexpr decay_t<T> operator()(T&& val) noexcept(
        is_nothrow_constructible_v<decay_t<T>, T>) {
        return __DPL forward<T>(val);
    }

    template <canonical_vector T>
    requires fixed_width_abi<simd_abi_type_t<T>> &&
        (sizeof(T) == sizeof(rebind_simd_t<T, ToE>))
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr rebind_simd_t<T, ToE> operator()(T from) noexcept {
        using To DPL_NODEBUG = rebind_simd_t<T, ToE>;
        using to_vector DPL_NODEBUG =
            typename simd_abi_traits<To>::native_vector;
        using from_vector DPL_NODEBUG =
            typename simd_abi_traits<T>::native_vector;
        if constexpr (same_as<to_vector, from_vector>) {
            return +from;
        } else if consteval {
            // Workaround MSVC's unions
            array_for<T> buffer;
            dx::store(from, buffer.data);
            return dx::load<To>( __DPL bit_cast<array_for<To>>(buffer).data);
        } else {
            return __DPL bit_cast<To>(from);
        }
    }

    template <canonical_vector T>
    requires common_size_with<simd_element_type_t<T>, ToE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr rebind_simd_t<T, ToE> operator()(T from) noexcept {
        using To = rebind_simd_t<T, ToE>;
        using to_mask = typename simd_abi_traits<To>::native_mask;
        using from_mask = typename simd_abi_traits<T>::native_mask;
        constexpr auto width = typename simd_abi_traits<T>::size();
        if constexpr (same_as<to_mask, from_mask>) {
            return +from;
        } else if consteval {
            // Workaround MSVC's unions
            return [&]<size_t... Is>(index_sequence<Is...>) {
                return dx::from_bitset<To>(bitset<width>(from[imm<Is>]...));
            }(iota_sequence<T>);
        } else {
            return __DPL bit_cast<To>(from);
        }
    }
};

template <typename ToE>
struct canonical_impl<reinterpret_t<ToE>> {
    template <canonical_vector T>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
    static constexpr rebind_simd_t<T, ToE> operator()(T val) noexcept
    requires requires { reinterpret<ToE>(internal::abi<T>, val); }
    {
        return reinterpret<ToE>(internal::abi<T>, val);
    }

    template <canonical_mask T>
    requires common_size_with<simd_element_type_t<T>, ToE>
    DPL_ATTRIBUTES(_HIDE_FROM_ABI, CONST, NODISCARD)
    static constexpr rebind_simd_t<T, ToE> operator()(T val) noexcept
    requires requires { reinterpret<ToE>(internal::abi<T>, val); }
    {
        return reinterpret<ToE>(internal::abi<T>, val);
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
template <typename To>
inline constexpr internal::reinterpret_t<To> reinterpret{};
} // namespace cpo
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
