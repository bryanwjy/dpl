// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_size_with.h"
#include "dpl/core/concepts/simd_element.h"
#include "dpl/core/concepts/simd_mask_type.h"
#include "dpl/core/concepts/simd_type.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
DPL_EXPORT template <typename M, typename T>
concept simd_mask_for = simd_mask_type<M> && simd_type<T> &&
    common_abi_with<typename T::abi_type, typename M::abi_type> &&
    common_size_with<simd_element_type_t<M>, simd_element_type_t<T>>;
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
