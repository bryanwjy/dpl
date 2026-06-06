// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/simd_abi_type.h"
#include "dpl/core/type_traits/simd_element_representation.h"
#include "dpl/core/type_traits/simd_element_type.h"
#include "dpl/core/type_traits/simd_native_type.h"
#include "dpl/core/type_traits/simd_value_type.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/remove_cv.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename>
struct simd_traits {};

DPL_EXPORT template <typename T>
requires enable_simd_mask<remove_cv_t<T>> || enable_simd_vector<remove_cv_t<T>>
struct simd_traits<T> {
    using type = remove_cv_t<T>;
    using abi_type = simd_abi_type_t<T>;
    using value_type = simd_value_type_t<T>;
    using element_type = simd_element_type_t<T>;
    using element_representation =
        simd_element_representation_t<abi_type, element_type>;
    using native_type = simd_native_type_t<T>;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
