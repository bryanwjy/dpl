// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_type.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <typename T>
concept simd_mask = simd_type<T> && atom::simd_mask<remove_cv_t<T>>;

DPL_EXPORT template <typename T>
concept scalable_mask = simd_mask<T> && scalable_simd_type<T>;

DPL_EXPORT template <typename T>
concept fixed_width_mask = simd_mask<T> && fixed_width_simd_type<T>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
