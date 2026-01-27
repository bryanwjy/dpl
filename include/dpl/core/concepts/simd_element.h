// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

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

DPL_EXPORT template <typename T>
struct simd_element_type {};

DPL_EXPORT template <typename T>
using simd_element_type_t = typename simd_element_type<T>::type;

DPL_EXPORT template <simd_element T>
struct simd_element_type<T> {
    using type DPL_NODEBUG = T;
};
} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
