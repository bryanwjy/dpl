// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_traits/details/simd_base.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/derived_from.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename T>
struct simd_abi_base : internal::simd_base {
    __DPL_HIDE_FROM_ABI constexpr ~simd_abi_base() = default;
};

template <typename T>
inline constexpr bool enable_simd_abi =
    (derived_from<T, simd_abi_base<T>> && derived_from<T, internal::simd_base>);

} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
