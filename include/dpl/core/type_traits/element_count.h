// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_element.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename, typename...>
inline constexpr size_t element_count = 0;

DPL_EXPORT template <simd_class T>
inline constexpr size_t element_count<T> =
    T::abi_type::size / sizeof(simd_element_type_t<T>);

DPL_EXPORT template <simd_element E, simd_abi A>
inline constexpr size_t element_count<E, A> = A::size / sizeof(E);
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
