// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#  include "dpl/core/type_interface/enable_const_mask.h"
#  include "dpl/core/type_interface/enable_simd_abi.h"
#  include "dpl/core/type_interface/enable_simd_mask.h"
#  include "dpl/core/type_interface/enable_simd_vector.h"
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/bit/char_bit.h"
#  include "dpl/std/concepts/different_from.h"
#  include "dpl/std/type_traits/conditional.h"
#  include "dpl/std/type_traits/is_function.h"
#  include "dpl/std/type_traits/is_object.h"
#  include "dpl/std/type_traits/is_trivially_copyable.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#  include "dpl/std/utility/bitset.h"
#  include "dpl/std/utility/ignore.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
DPL_EXPORT namespace datapar::internal {

// Cheaper declval, assumes T is always referenceable
template <typename T>
T&& declarg() noexcept;

// Cheaper than invoke_result_t, avoids instantiating __DPL invoke & friends
template <typename D, typename... Ts>
using cpo_result_t DPL_NODEBUG =
    decltype(internal::declarg<D const&>()(internal::declarg<Ts>()...));

template <typename D, typename... Ts>
struct cpo_result {
    using type DPL_NODEBUG = cpo_result_t<D, Ts...>;
};

} // namespace datapar::internal

DPL_EXPORT namespace datapar {

template <typename T>
struct simd_abi_type;
template <typename T>
struct simd_element_type;
template <typename T>
struct simd_vector_type;
template <typename T>
struct simd_mask_type;
template <typename A, typename E>
struct simd_element_representation;
template <typename>
struct canonical_type;
template <typename, typename = __DPL ignore_t>
struct simd_abi_traits;

namespace internal {
template <typename E, typename A>
concept has_representation_for = requires {
    typename simd_element_representation<A, E>::type;
    typename A::template native_vector<E>;
    typename A::template native_mask<E>;
};

template <typename T>
concept has_simd_abi = requires { typename simd_abi_type<T>::type; };

template <typename T>
concept has_simd_element = requires { typename simd_element_type<T>::type; };

template <typename T>
concept has_simd_members = has_simd_element<T> && has_simd_abi<T>;

template <typename T>
concept has_expression_result =
    (enable_simd_mask<T> || enable_simd_vector<T>) &&
    different_from<typename canonical_type<T>::type, T> &&
    requires { typename T::result_type; } &&
    different_from<typename T::result_type, T> &&
    (enable_simd_mask<typename T::result_type> ||
        enable_simd_vector<typename T::result_type>);

template <typename T>
concept none_abi_type = has_simd_abi<T> && !enable_simd_abi<T>;

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

} // namespace internal

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
