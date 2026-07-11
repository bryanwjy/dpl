// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/utility/ignore.h"
#endif

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
template <typename, typename = __DPL ignore_t>
struct simd_abi_traits;
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END

// IWYU pragma: end_exports
