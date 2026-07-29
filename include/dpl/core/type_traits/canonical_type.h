// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/details/has_simd_members.h"
#include "dpl/core/type_traits/enable_simd_mask.h"
#include "dpl/core/type_traits/enable_simd_vector.h"
#include "dpl/core/type_traits/simd_abi_traits.h"
#include "dpl/core/type_traits/simd_abi_type.h"
#include "dpl/core/type_traits/simd_element_type.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename E, typename A>
struct make_canonical_vector {};

template <typename E, typename A>
using make_canonical_vector_t = typename make_canonical_vector<E, A>::type;

template <typename E, internal::abi_with_fixed_size<E> A>
struct make_canonical_vector<E, A> {
    using type = basic_vector<E, A>;
};

/**
 * Scalable ABI workaround
 *
 * Since scalable backend types cannot be encapsulated,
 * the native types need to be declared canonical
 */
template <typename E, typename A>
requires internal::has_representation_for<E, A> &&
    (!internal::abi_with_fixed_size<A, E>)
struct make_canonical_vector<E, A> {
    using type = typename simd_abi_traits<A, E>::native_vector;
};

template <typename E, typename A>
struct make_canonical_mask {};

template <typename E, typename A>
using make_canonical_mask_t = typename make_canonical_mask<E, A>::type;

template <typename E, internal::abi_with_fixed_size<E> A>
struct make_canonical_mask<E, A> {
    using type = basic_mask<E, A>;
};

/**
 * Scalable ABI workaround
 *
 * Since scalable backend types cannot be encapsulated,
 * the native types need to be declared canonical
 */
template <typename E, typename A>
requires internal::has_representation_for<E, A> &&
    (!internal::abi_with_fixed_size<A, E>)
struct make_canonical_mask<E, A> {
    using type = typename simd_abi_traits<A, E>::native_mask;
};

template <typename T>
struct canonical_type {};

template <typename T>
struct canonical_type<T const> : canonical_type<T> {};
template <typename T>
struct canonical_type<T volatile> : canonical_type<T> {};
template <typename T>
struct canonical_type<T const volatile> : canonical_type<T> {};
template <typename T>
struct canonical_type<T&> : canonical_type<T> {};
template <typename T>
struct canonical_type<T&&> : canonical_type<T> {};

template <typename T>
using canonical_type_t = typename canonical_type<T>::type;

template <typename T>
requires enable_simd_vector<T> && internal::has_simd_members<T>
struct canonical_type<T> :
    make_canonical_vector<simd_element_type_t<T>, simd_abi_type_t<T>> {};

template <typename T>
requires enable_simd_mask<T> && internal::has_simd_members<T>
struct canonical_type<T> :
    make_canonical_mask<simd_element_type_t<T>, simd_abi_type_t<T>> {};

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
