// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/derived_from.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <typename T>
struct const_mask_base {
    __DPL_HIDE_FROM_ABI constexpr ~const_mask_base() = default;
};

template <typename T>
inline constexpr bool enable_const_mask = derived_from<T, const_mask_base<T>>;
} // namespace datapar

__DPL_DEFAULT_NAMESPACE_END
