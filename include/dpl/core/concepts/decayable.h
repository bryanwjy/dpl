
// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/basic_type.h"
#include "dpl/core/concepts/simd_traits.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

template <typename T, operation_category C>
concept decayable_vector_for =
    extended_vector<T> && ((simd_traits<T>::decay_policy & C) == C);
template <typename T, operation_category C>
concept decayable_mask_for =
    extended_mask<T> && ((simd_traits<T>::decay_policy & C) == C);
template <typename T, operation_category C>
concept decayable_simd_for =
    extended_class<T> && ((simd_traits<T>::decay_policy & C) == C);

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
