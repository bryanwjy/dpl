// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/simd_abi.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <typename T>
concept simd_element =
    !same_as<T, bool> && (integral<T> || floating_point<T> || enumeration<T>);

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
