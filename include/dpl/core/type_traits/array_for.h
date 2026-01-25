// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/element_count.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_element.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename E, typename... A>
struct array_for {};

template <simd_element E, simd_abi A>
struct alignas(A::alignment) array_for<E, A> {
    E data[A::size / sizeof(E)];
};

template <simd_type T>
struct alignas(T::abi_type::alignment) array_for<T> {
    simd_element_type_t<T> data[element_count<T>];
};

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
