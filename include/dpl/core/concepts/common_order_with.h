// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/concepts/common_abi_with.h"
#include "dpl/core/concepts/common_bits_with.h"
#include "dpl/core/concepts/common_float_with.h"
#include "dpl/core/concepts/common_integral_with.h"

#if !DPL_MODULES
#  include "dpl/std/concepts/same_as.h"
#  include "dpl/std/concepts/totally_ordered.h"
#  include "dpl/std/type_traits/is_enum.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <typename A, typename B>
concept common_order_with = (common_bits_with<A, B> &&
    (same_as<A, B> || common_float_with<A, B> || common_integral_with<A, B>));

DPL_EXPORT template <typename A, typename B>
concept simd_common_order_with = simd_common_abi_with<A, B> &&
    common_order_with<simd_element_type_t<A>, simd_element_type_t<B>>;

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
