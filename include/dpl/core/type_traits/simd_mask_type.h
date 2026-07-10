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
struct simd_mask_type {};
DPL_EXPORT template <typename T>
struct simd_mask_type<T const> : simd_mask_type<T> {};
DPL_EXPORT template <typename T>
struct simd_mask_type<T volatile> : simd_mask_type<T> {};
DPL_EXPORT template <typename T>
struct simd_mask_type<T const volatile> : simd_mask_type<T> {};
DPL_EXPORT template <typename T>
struct simd_mask_type<T&> : simd_mask_type<T> {};
DPL_EXPORT template <typename T>
struct simd_mask_type<T&&> : simd_mask_type<T> {};

DPL_EXPORT template <typename T>
using simd_mask_type_t = typename simd_mask_type<T>::type;

DPL_EXPORT template <typename T>
requires enable_simd_vector<T> && requires {
    typename T::mask_type;
    requires enable_simd_mask<typename T::mask_type>;
}
struct simd_mask_type<T> {
    using type DPL_NODEBUG = typename T::mask_type;
};

DPL_EXPORT template <typename T>
requires enable_simd_vector<T> && internal::has_simd_members<T>
struct simd_mask_type<T> {
    using type DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;
};

DPL_EXPORT template <typename T>
requires enable_simd_mask<T>
struct simd_mask_type<T> {
    using type DPL_NODEBUG = T;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
