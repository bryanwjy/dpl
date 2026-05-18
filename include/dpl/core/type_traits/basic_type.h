// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_element_representation.h"
#  include "dpl/core/concepts/simd_mask.h"
#  include "dpl/core/concepts/simd_vector.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT template <typename T>
struct canonical_type {};

DPL_EXPORT template <typename T>
using canonical_type_t = typename canonical_type<T>::type;

DPL_EXPORT template <simd_vector T>
struct canonical_type<T> {
    using type DPL_NODEBUG =
        basic_vector<simd_element_representation_t<typename T::abi_type,
                         typename T::value_type>,
            typename T::abi_type>;
};

DPL_EXPORT template <simd_mask T>
struct canonical_type<T> {
    using type DPL_NODEBUG =
        basic_mask<simd_element_representation_t<typename T::abi_type,
                       typename T::simd_vector::value_type>,
            typename T::abi_type>;
};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
