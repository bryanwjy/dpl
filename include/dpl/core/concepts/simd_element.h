// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/floating_point.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/concepts/regular.h"
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/concepts/totally_ordered.h"
#  include "dpl/std/type_traits/is_enum.h"
#  include "dpl/std/type_traits/is_trivially_copyable.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN

namespace datapar {

DPL_EXPORT template <typename T>
concept simd_element =
    !same_as<T, bool> && (integral<T> || floating_point<T> || is_enum_v<T>);

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
