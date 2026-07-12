// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_type.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar::internal {
template <typename A>
requires simd_abi<A> || simd_type<A>
inline constexpr simd_abi_type_t<A> abi{};
} // namespace datapar::internal

__DPL_DEFAULT_NAMESPACE_END
