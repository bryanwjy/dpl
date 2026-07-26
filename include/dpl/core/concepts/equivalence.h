// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_simd_type_with.h"
#include "dpl/core/concepts/common_size_with.h"
#include "dpl/core/concepts/simd_mask.h"
#include "dpl/core/concepts/simd_vector.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename T, typename A>
concept simd_type_with_common_abi =
    simd_type<T> && simd_abi<A> && common_abi_with<simd_abi_type_t<T>, A>;

template <typename T, typename A>
concept simd_type_with_abi =
    simd_type_with_common_abi<T, A> && same_as<simd_abi_type_t<T>, A>;

template <typename T, typename A>
concept vector_with_common_abi =
    simd_type_with_common_abi<T, A> && simd_vector<T>;

template <typename T, typename A>
concept vector_with_abi =
    vector_with_common_abi<T, A> && same_as<simd_abi_type_t<T>, A>;

template <typename T, typename A>
concept mask_with_common_abi = simd_type_with_common_abi<T, A> && simd_mask<T>;

template <typename T, typename A>
concept mask_with_abi =
    mask_with_common_abi<T, A> && same_as<simd_abi_type_t<T>, A>;

template <typename T, typename E, typename A = simd_abi_type_t<T>>
concept simd_type_with =
    simd_type_with_abi<T, A> && same_as<simd_element_type_t<T>, E>;

template <typename T, typename E, typename A = simd_abi_type_t<T>>
concept mask_with = simd_type_with<T, E, A> && simd_mask<T>;

template <typename T, typename E, typename A = simd_abi_type_t<T>>
concept vector_with = simd_type_with<T, E, A> && simd_vector<T>;

namespace atom {
template <typename T, typename U>
concept equivalent_mask_elements = simd_mask<T> && simd_mask<U> &&
    common_size_with<simd_element_type_t<T>, simd_element_type_t<U>>;

template <typename T, typename U>
concept equivalent_vector_elements = simd_vector<T> && simd_vector<U> &&
    same_as<simd_element_type_t<T>, simd_element_type_t<U>>;
} // namespace atom

template <typename T, typename U>
concept common_vector_with = common_simd_type_with<T, U> && simd_vector<T> &&
    simd_vector<U> && atom::equivalent_vector_elements<T, U>;

template <typename T, typename U>
concept vector_subsumed_by = common_vector_with<T, U> &&
    same_abi_as<common_abi_t<T, U>, simd_abi_type_t<U>>;

template <typename T, typename U>
concept common_mask_with = common_simd_type_with<T, U> && simd_mask<T> &&
    simd_mask<U> && atom::equivalent_mask_elements<T, U>;

template <typename T, typename U>
concept equivalent_simd_type_with =
    (common_vector_with<T, U> || common_mask_with<T, U>) &&
    same_as<simd_abi_type_t<T>, simd_abi_type_t<U>>;

template <typename T, typename U>
concept equivalent_vector_with =
    equivalent_simd_type_with<T, U> && atom::equivalent_vector_elements<T, U>;

template <typename T, typename U>
concept equivalent_mask_with =
    equivalent_simd_type_with<T, U> && atom::equivalent_mask_elements<T, U>;

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
