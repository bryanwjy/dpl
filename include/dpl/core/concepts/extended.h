// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/canonical.h"
#include "dpl/core/concepts/simd_mask.h"
#include "dpl/core/concepts/simd_type.h"
#include "dpl/core/concepts/simd_vector.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <typename T>
concept extended_simd_type =
    simd_type<remove_cvref_t<T>> && !canonical_simd_type<remove_cvref_t<T>>;

DPL_EXPORT template <typename T>
concept extended_vector =
    simd_vector<remove_cvref_t<T>> && extended_simd_type<T>;

DPL_EXPORT template <typename T>
concept extended_mask = simd_mask<remove_cvref_t<T>> && extended_simd_type<T>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
