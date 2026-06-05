// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/derived_from.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
struct const_mask_base {
    __DPL_HIDE_FROM_ABI constexpr ~const_mask_base() = default;
};

DPL_EXPORT template <typename T>
inline constexpr bool enable_const_mask = derived_from<T, const_mask_base<T>>;

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
