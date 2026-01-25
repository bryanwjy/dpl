// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/basic_type.h"
#include "dpl/core/concepts/simd_element.h"

#if !DPL_MODULE
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T>
concept arithmetic_type =
    !same_as<T, bool> && (integral<T> || floating_point<T>);

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
