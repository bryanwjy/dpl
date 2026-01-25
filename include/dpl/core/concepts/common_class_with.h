// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_class.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename A, typename B>
concept common_class_with =
    simd_class<A> && simd_class<B> && (simd_type<A> && simd_type<B>) ||
    (simd_mask_type<A> && simd_mask_type<B>);

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
