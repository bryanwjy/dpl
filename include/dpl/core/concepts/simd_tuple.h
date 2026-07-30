// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/details/cpo_invocable.h"
#  include "dpl/core/type_traits/enable_simd_tuple.h"
#  include "dpl/core/type_traits/simd_native_type.h"
#  include "dpl/std/concepts/tuple_like.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
namespace internal {
template <size_t I>
struct get_element_t;
template <size_t I>
struct set_element_t;
} // namespace internal

namespace atom {
template <typename T, size_t I>
concept simd_tuple_accessible =
    __DPL details::concepts::has_tuple_element<T, I> &&
    internal::cpo_invocable<internal::get_element_t<I>, T> &&
    internal::cpo_invocable<internal::set_element_t<I>, T&,
        std::tuple_element_t<I, T>>;

template <typename T, typename S = __DPL details::concepts::sequence_for_t<T>>
inline constexpr bool is_simd_tuple_like = false;
template <typename T, size_t... Is>
inline constexpr bool is_simd_tuple_like<T, index_sequence<Is...>> =
    (... && simd_tuple_accessible<T, Is>) &&
    (... && same_as<std::tuple_element_t<0, T>, std::tuple_element_t<Is, T>>);

template <typename T>
concept has_native_tuple = requires { typename simd_native_type_t<T>; };

// Should we support mask tuples?
// SVE has it, but I can't imagine there being much use for them
template <typename T>
concept simd_tuple =
    enable_simd_tuple<T> && requires { typename simd_abi_type_t<T>; } &&
    (!has_native_tuple<T> ||
        explicitly_convertible_to<T, simd_native_type_t<T>>) &&
    atom::is_simd_tuple_like<T>;
} // namespace atom

template <typename T>
concept simd_tuple = atom::simd_tuple<remove_cvref_t<T>> &&
    requires {
        // The following should not be specialized for tuples
        // and are derived from tuple_element_t
        typename simd_element_type_t<T>;
        typename simd_value_type_t<T>;
        typename simd_vector_type_t<T>;
    } && same_as<simd_element_type_t<T>, simd_vector_type_t<T>> &&
    same_as<simd_abi_type_t<simd_element_type_t<T>>, simd_abi_type_t<T>> &&
    same_as<simd_value_type_t<simd_element_type_t<T>>, simd_value_type_t<T>>;

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
