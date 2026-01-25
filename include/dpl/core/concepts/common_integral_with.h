// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/fwd.h"

#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_basic_element_with.h"
#include "dpl/core/concepts/simd_element.h"

#if !DPL_MODULES
#  include "dpl/std/bit/bit_cast.h"
#  include "dpl/std/concepts/floating_point.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename T, typename U>
concept common_integral_with =
    sizeof(T) == sizeof(U) && integral<T> && integral<U> &&
    ((signed_integral<T> && signed_integral<U>) ||
        (unsigned_integral<T> && unsigned_integral<U>)) &&
    internal::common_basic_element_with<T, U>;

DPL_EXPORT template <typename A, typename B>
concept simd_common_integral_with = simd_common_abi_with<A, B> &&
    common_integral_with<simd_element_type_t<A>, simd_element_type_t<B>>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
