// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_class_with.h"
#include "dpl/core/concepts/common_size_with.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace atom {
template <typename T, typename U>
concept common_float_with = (brain_float<T> && brain_float<U>) ||
    (!brain_float<T> && !brain_float<U> && floating_point<T> &&
        floating_point<U>);
}

DPL_EXPORT template <typename T, typename U>
concept common_float_with =
    common_size_with<T, U> && atom::common_float_with<T, U>;

DPL_EXPORT template <typename A, typename B>
concept common_float_simd_with = common_class_with<A, B> &&
    common_float_with<simd_lane_representation_t<A>,
        simd_lane_representation_t<B>>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
