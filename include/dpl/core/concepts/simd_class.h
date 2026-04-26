// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_mask_type.h"
#include "dpl/core/concepts/simd_type.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename T>
concept simd_class =
    atom::simd_basics<T> && (atom::simd_mask_type<T> || atom::simd_type<T>);

DPL_EXPORT template <typename T>
concept fixed_width_class =
    simd_class<T> && fixed_width_abi<typename T::abi_type>;

DPL_EXPORT template <typename T>
concept scalable_class = simd_class<T> && scalable_abi<typename T::abi_type>;
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
