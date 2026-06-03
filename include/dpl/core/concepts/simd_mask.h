// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"
#include "dpl/core/concepts/simd_basics.h"
#include "dpl/core/concepts/simd_class.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <typename T>
concept simd_mask = simd_class<T> && atom::simd_mask<remove_cv_t<T>>;

DPL_EXPORT template <typename T>
concept scalable_mask =
    simd_mask<T> && scalable_abi<typename remove_cv_t<T>::abi_type>;
DPL_EXPORT template <typename T>
concept fixed_width_mask =
    simd_mask<T> && fixed_width_abi<typename remove_cv_t<T>::abi_type>;
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
