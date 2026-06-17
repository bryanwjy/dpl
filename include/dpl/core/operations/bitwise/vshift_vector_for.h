// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/canonical.h"
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/type_traits/simd_abi_type.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {

template <typename T, typename L>
concept vshift_vector_for =
    simd_vector<T> && integral<simd_element_type_t<T>> &&
    common_abi_with<simd_abi_type_t<L>, simd_abi_type_t<T>>;

template <typename T, typename L>
concept canonical_vshift_vector_for =
    vshift_vector_for<T, L> && canonical_vector<T> &&
    common_size_with<simd_element_type_t<L>, simd_element_type_t<T>>;

} // namespace datapar::internal
DPL_DEFAULT_NAMESPACE_END
