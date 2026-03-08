// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_mask_type.h"
#include "dpl/core/concepts/simd_type.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
concept simd_class = simd_mask_type<T> || simd_type<T>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
