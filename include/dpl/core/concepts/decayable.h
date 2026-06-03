
// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/basic_type.h"
#include "dpl/core/concepts/simd_traits.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
template <typename T, operation_category C>
concept decayable_simd_for = canonical_class<T> ||
    (extended_class<T> && ((simd_traits<T>::decay_policy & C) == C));

template <typename T, operation_category C>
concept decayable_vector_for = decayable_simd_for<T, C> && simd_vector<T>;

template <typename T, operation_category C>
concept decayable_mask_for = decayable_simd_for<T, C> && simd_mask<T>;

template <operation_category C, typename... T>
concept all_decayable = (... && decayable_simd_for<T, C>);

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
