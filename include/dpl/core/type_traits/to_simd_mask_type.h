// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/simd_element_type.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_class.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct to_simd_mask_type {};

DPL_EXPORT template <simd_class T>
using to_simd_mask_type_t = typename to_simd_mask_type<T>::type;

DPL_EXPORT template <simd_class T>
struct to_simd_mask_type<T> {
    using type DPL_NODEBUG =
        simd_mask<simd_element_type_t<T>, typename T::abi_type>;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
