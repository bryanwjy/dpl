// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/type_interface/simd_base.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#  include "dpl/std/concepts/derived_from.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct simd_mask_base : simd_base {
    __DPL_HIDE_FROM_ABI constexpr ~simd_mask_base() = default;
};

DPL_EXPORT template <typename T>
inline constexpr bool enable_simd_mask =
    (derived_from<T, simd_mask_base<T>> && derived_from<T, simd_base>);
DPL_EXPORT template <typename E, typename A>
inline constexpr bool enable_simd_mask<basic_mask<E, A>> = true;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
