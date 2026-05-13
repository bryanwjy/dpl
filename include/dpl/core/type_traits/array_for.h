// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/simd_abi_traits.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename E, typename... A>
struct array_for {};

template <simd_element E, fixed_width_abi A>
struct alignas(simd_abi_traits<E, A>::alignment) array_for<E, A> {
    E data[simd_abi_traits<E, A>::size];
};

template <simd_type T>
struct alignas(simd_abi_traits<T>::alignment) array_for<T> {
    simd_lane_type_t<T> data[simd_abi_traits<T>::size];
};

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
