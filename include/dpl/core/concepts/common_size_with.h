// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_class_with.h"

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
DPL_EXPORT template <typename T, typename U>
concept common_size_with = sizeof(T) == sizeof(U);

DPL_EXPORT template <typename T, typename U>
concept common_size_simd_with =
    common_abi_simd_with<T, U> && common_class_with<T, U> &&
    common_size_with<simd_element_type_t<T>, simd_element_type_t<U>>;
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
