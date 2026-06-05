// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#if !DPL_MODULES
#  include "dpl/core/concepts/common_simd_type_with.h"
#  include "dpl/core/concepts/common_size_with.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_mask.h"
#  include "dpl/core/type_traits/common_abi.h"
#  include "dpl/core/type_traits/simd_element_type.h"
#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename T, typename L, typename R, typename A = common_abi_t<L, R>>
concept canonical_bitwise_result = vector_with<T, simd_element_type_t<L>, A> &&
    vector_with<T, simd_element_type_t<R>, A>;

template <typename T, typename L, typename R, typename A = common_abi_t<L, R>>
concept canonical_bitwise_mask = simd_mask<T> &&
    common_size_with<simd_element_type_t<L>, simd_element_type_t<T>> &&
    common_size_with<simd_element_type_t<R>, simd_element_type_t<T>> &&
    same_abi_as<A, typename T::abi_type>;

template <typename T, typename In, typename A = typename In::abi_type>
concept canonical_bitshift_result = common_simd_type_with<T, In> &&
    same_as<simd_element_type_t<T>, simd_element_type_t<In>> &&
    same_abi_as<A, typename T::abi_type>;

} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
