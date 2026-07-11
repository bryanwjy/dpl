// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/details/has_simd_members.h"
#include "dpl/core/type_traits/enable_simd_mask.h"
#include "dpl/core/type_traits/enable_simd_vector.h"
#include "dpl/core/type_traits/simd_abi_type.h"
#include "dpl/core/type_traits/simd_element_type.h"

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename T>
struct simd_mask_type {};
template <typename T>
struct simd_mask_type<T const> : simd_mask_type<T> {};
template <typename T>
struct simd_mask_type<T volatile> : simd_mask_type<T> {};
template <typename T>
struct simd_mask_type<T const volatile> : simd_mask_type<T> {};
template <typename T>
struct simd_mask_type<T&> : simd_mask_type<T> {};
template <typename T>
struct simd_mask_type<T&&> : simd_mask_type<T> {};

template <typename T>
using simd_mask_type_t = typename simd_mask_type<T>::type;

template <typename T>
requires enable_simd_vector<T> && requires {
    typename T::mask_type;
    requires enable_simd_mask<typename T::mask_type>;
}
struct simd_mask_type<T> {
    using type DPL_NODEBUG = typename T::mask_type;
};

template <typename T>
requires enable_simd_vector<T> && internal::has_simd_members<T>
struct simd_mask_type<T> {
    using type DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;
};

template <typename T>
requires enable_simd_mask<T>
struct simd_mask_type<T> {
    using type DPL_NODEBUG = T;
};

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
