// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

// IWYU pragma: begin_exports
__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
template <typename T>
struct simd_abi_type;
template <typename T>
struct simd_element_type;
template <typename T>
struct simd_vector_type;
template <typename T>
struct simd_mask_type;
template <typename A, typename E>
struct simd_element_representation;
template <typename>
struct canonical_type;
template <typename, typename = void>
struct simd_abi_traits;
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

// IWYU pragma: end_exports
