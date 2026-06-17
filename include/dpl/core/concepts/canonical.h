// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_mask.h"
#include "dpl/core/concepts/simd_type.h"
#include "dpl/core/concepts/simd_vector.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/is_canonical_type.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_element_representation.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#  include "dpl/std/type_traits/remove_cvref.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

namespace atom {
template <typename T>
concept canonical_type = is_canonical_type_v<T> &&
    same_as<simd_element_representation_t<simd_abi_type_t<T>,
                simd_element_type_t<T>>,
        simd_element_type_t<T>>;
} // namespace atom

DPL_EXPORT template <typename T>
concept canonical_simd_type =
    simd_type<T> && atom::canonical_type<remove_cvref_t<T>>;

DPL_EXPORT template <typename T>
concept canonical_vector = simd_vector<T> && canonical_simd_type<T>;

DPL_EXPORT template <typename T>
concept canonical_mask = simd_mask<T> && canonical_simd_type<T>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
