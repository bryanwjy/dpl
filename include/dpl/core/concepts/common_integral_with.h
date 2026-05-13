// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_size_with.h"
#include "dpl/core/concepts/simd_element.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

namespace atom {
template <typename A, typename B>
concept common_integral_with = integral<A> && integral<B> &&
    ((signed_integral<A> && signed_integral<B>) ||
        (unsigned_integral<A> && unsigned_integral<B>));
}

DPL_EXPORT template <typename A, typename B>
concept common_integral_with =
    common_size_with<A, B> && atom::common_integral_with<A, B>;

DPL_EXPORT template <typename A, typename B>
concept common_integral_simd_with = common_class_with<A, B> &&
    common_integral_with<simd_lane_representation_t<A>,
        simd_lane_representation_t<B>>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
