// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_basics.h"
#include "dpl/core/concepts/simd_class.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <typename T>
concept simd_vector = simd_class<T> && atom::simd_vector<remove_cv_t<T>>;

DPL_EXPORT template <typename T>
concept scalable_vector =
    simd_vector<T> && scalable_abi<typename remove_cv_t<T>::abi_type>;
DPL_EXPORT template <typename T>
concept fixed_width_vector =
    simd_vector<T> && fixed_width_abi<typename remove_cv_t<T>::abi_type>;
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
