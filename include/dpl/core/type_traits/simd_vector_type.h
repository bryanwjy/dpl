// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/simd_abi_type.h"
#include "dpl/core/type_traits/simd_element_type.h"

#if !DPL_MODULES
#  include "dpl/core/details/type_traits.h"
#  include "dpl/core/type_interface/enable_simd_mask.h"
#  include "dpl/core/type_interface/enable_simd_vector.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct simd_vector_type {};
DPL_EXPORT template <typename T>
struct simd_vector_type<T const> : simd_vector_type<T> {};
DPL_EXPORT template <typename T>
struct simd_vector_type<T volatile> : simd_vector_type<T> {};
DPL_EXPORT template <typename T>
struct simd_vector_type<T const volatile> : simd_vector_type<T> {};
DPL_EXPORT template <typename T>
struct simd_vector_type<T&> : simd_vector_type<T> {};
DPL_EXPORT template <typename T>
struct simd_vector_type<T&&> : simd_vector_type<T> {};

DPL_EXPORT template <typename T>
using simd_vector_type_t = typename simd_vector_type<T>::type;

DPL_EXPORT template <typename T>
requires enable_simd_vector<T>
struct simd_vector_type<T> {
    using type DPL_NODEBUG = T;
};

DPL_EXPORT template <typename T>
requires enable_simd_mask<T> && requires {
    typename T::vector_type;
    requires enable_simd_vector<typename T::vector_type>;
}
struct simd_vector_type<T> {
    using type DPL_NODEBUG = typename T::vector_type;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
