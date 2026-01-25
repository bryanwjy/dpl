// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/basic_element.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT template <typename T>
struct basic_type {};

DPL_EXPORT template <typename T>
using basic_type_t = typename basic_type<T>::type;

DPL_EXPORT template <simd_type T>
struct basic_type<T> {
    using type DPL_NODEBUG = basic_simd<basic_element_t<typename T::value_type>,
        typename T::abi_type>;
};

DPL_EXPORT template <simd_mask_type T>
struct basic_type<T> {
    using type DPL_NODEBUG =
        simd_mask<basic_element_t<typename T::simd_type::value_type>,
            typename T::abi_type>;
};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
