// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_lane_type.h"
#  include "dpl/core/concepts/simd_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar::internal {
template <typename T>
concept integral_simd = simd_type<T> && integral<simd_lane_type_t<T>>;
} // namespace datapar::internal

DPL_DEFAULT_NAMESPACE_END
