// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/simd_abi_type.h"
#include "dpl/core/type_traits/simd_element_type.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#  include "dpl/core/type_interface/enable_simd_mask.h"
#  include "dpl/core/type_interface/enable_simd_vector.h"
#  include "dpl/core/type_traits/details/has_simd_members.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct canonical_type {};

DPL_EXPORT template <typename T>
struct canonical_type<T const> : canonical_type<T> {};
DPL_EXPORT template <typename T>
struct canonical_type<T volatile> : canonical_type<T> {};
DPL_EXPORT template <typename T>
struct canonical_type<T const volatile> : canonical_type<T> {};
DPL_EXPORT template <typename T>
struct canonical_type<T&> : canonical_type<T> {};
DPL_EXPORT template <typename T>
struct canonical_type<T&&> : canonical_type<T> {};

DPL_EXPORT template <typename T>
using canonical_type_t = typename canonical_type<T>::type;

DPL_EXPORT template <typename T>
requires enable_simd_vector<T> && internal::has_simd_members<T>
struct canonical_type<T> {
    using type DPL_NODEBUG =
        basic_vector<simd_element_type_t<T>, simd_abi_type_t<T>>;
};

DPL_EXPORT template <typename T>
requires enable_simd_mask<T> && internal::has_simd_members<T>
struct canonical_type<T> {
    using type DPL_NODEBUG =
        basic_mask<simd_element_type_t<T>, simd_abi_type_t<T>>;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
