// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_size_with.h"
#include "dpl/core/concepts/simd_mask.h"

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {
template <typename M, typename T>
concept compatible_mask_with = simd_mask<M> && simd_type<T> &&
    common_abi_with<simd_abi_type_t<M>, simd_abi_type_t<T>> &&
    common_size_with<simd_element_type_t<M>, simd_element_type_t<T>>;

template <typename M, typename T>
concept exact_mask_for = compatible_mask_with<M, T> &&
    same_abi_as<simd_abi_type_t<M>, simd_abi_type_t<T>>;
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
