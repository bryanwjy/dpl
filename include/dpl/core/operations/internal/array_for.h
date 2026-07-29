// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/type_traits/simd_abi_traits.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename E, typename = void>
struct array_for {};

template <typename E, fixed_width_abi A>
requires simd_element_for<E, A>
struct alignas(simd_abi_traits<E, A>::alignment) array_for<E, A> {
    E data[simd_abi_traits<E, A>::size];
};

template <simd_vector T>
struct alignas(simd_abi_traits<T>::alignment) array_for<T, void> {
    simd_element_type_t<T> data[simd_abi_traits<T>::size];
};

} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
