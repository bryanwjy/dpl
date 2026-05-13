// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_element.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT template <typename T>
struct basic_type {};

DPL_EXPORT template <typename T>
using basic_type_t = typename basic_type<T>::type;

DPL_EXPORT template <simd_type T>
struct basic_type<T> {
    using type DPL_NODEBUG =
        basic_simd<simd_element_representation_t<typename T::abi_type,
                       typename T::value_type>,
            typename T::abi_type>;
};

DPL_EXPORT template <simd_mask_type T>
struct basic_type<T> {
    using type DPL_NODEBUG =
        basic_simd_mask<simd_element_representation_t<typename T::abi_type,
                            typename T::simd_type::value_type>,
            typename T::abi_type>;
};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
