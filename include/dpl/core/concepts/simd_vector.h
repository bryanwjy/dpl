// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_type.h"

#if !DPL_MODULES
#  include "dpl/std/type_traits/remove_cvref.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <typename T>
concept simd_vector = simd_type<T> && atom::simd_vector<remove_cvref_t<T>>;

DPL_EXPORT template <typename T>
concept scalable_vector = simd_vector<T> && scalable_simd_type<T>;

DPL_EXPORT template <typename T>
concept fixed_width_vector = simd_vector<T> && fixed_width_simd_type<T>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
