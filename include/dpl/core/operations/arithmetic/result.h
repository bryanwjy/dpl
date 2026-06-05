// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: always_keep

#if !DPL_MODULES
#  include "dpl/core/concepts/common_abi_with.h"
#  include "dpl/core/concepts/equivalence.h"
#  include "dpl/core/concepts/simd_vector.h"
#  include "dpl/core/type_traits/common_abi.h"
#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {

template <typename T, typename L, typename R = L,
    typename A = common_abi_t<L, R>>
concept canonical_arithmetic_result =
    vector_with<T, simd_element_type_t<L>, A> &&
    vector_with<T, simd_element_type_t<R>, A>;

template <typename T, typename L, typename M, typename R,
    typename A = common_abi_t<L, M, R>>
concept canonical_fma_result =
    simd_vector<T> && same_as<typename L::value_type, typename T::value_type> &&
    same_as<typename M::value_type, typename T::value_type> &&
    same_as<typename R::value_type, typename T::value_type> &&
    same_abi_as<A, typename T::abi_type>;
} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
