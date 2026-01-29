// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_basic_element_with.h"
#include "dpl/core/concepts/common_size_with.h"
#include "dpl/core/concepts/simd_element.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/std/concepts/integral.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T, typename U>
concept common_integral_with =
    common_size_with<T, U> && integral<T> && integral<U> &&
    ((signed_integral<T> && signed_integral<U>) ||
        (unsigned_integral<T> && unsigned_integral<U>)) &&
    internal::common_basic_element_with<T, U>;

DPL_EXPORT template <typename A, typename B>
concept common_integral_simd_with = common_size_simd_with<A, B> &&
    common_integral_with<simd_element_type_t<A>, simd_element_type_t<B>>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
