// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd/basic.h"
#  include "dpl/std/concepts/derived_from.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT struct simd_base {
    __DPL_HIDE_FROM_ABI constexpr ~simd_base() = default;
};

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
