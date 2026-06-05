// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/simd_abi_traits.h"
#include "dpl/core/type_traits/simd_abi_type.h"
#include "dpl/core/type_traits/simd_element_representation.h"
#include "dpl/core/type_traits/simd_element_type.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/conditional.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename>
struct simd_traits {};

DPL_EXPORT template <typename T>
requires enable_simd_mask<T> || enable_simd_vector<T>
struct simd_traits<T> {
    using type = T;
    using abi_type = simd_abi_type_t<T>;
    using value_type = typename T::value_type;
    using element_type = simd_element_type_t<T>;
    using element_representation =
        simd_element_representation_t<abi_type, element_type>;
    using native_type = conditional_t<enable_simd_mask<T>,
        typename simd_abi_traits<abi_type, element_type>::native_mask,
        typename simd_abi_traits<abi_type, element_type>::native_vector>;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
