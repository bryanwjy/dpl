// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/details/has_simd_members.h"

#if !DPL_MODULES
#  include "dpl/core/type_traits/enable_simd_abi.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename T>
concept none_abi_type = has_simd_abi<T> && !enable_simd_abi<T>;
} // namespace datapar::internal
__DPL_DEFAULT_NAMESPACE_END
