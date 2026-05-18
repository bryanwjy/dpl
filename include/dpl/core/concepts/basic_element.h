// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/same_as.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

template <typename T>
concept basic_element = !same_as<T, bool> && (integral<T> || floating_point<T>);

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
