// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/fwd.h"

#  include "dpl/core/concepts/simd_abi.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/concepts/simd_element.h"
#  include "dpl/std/bit/popcount.h"
#  include "dpl/std/type_traits/constants.h"
#  include "dpl/std/utility/to_underlying.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT enum class operation_category : unsigned {
    lane_agnostic = 1u << 0,
    lane_reduction = 1u << 1,
    lane_permutation = 1u << 2,
    lane_conversion = 1u << 3,
    structural_transformation = 1u << 4,
    all = ((1u << 5) - 1),
    none = 0u,
};

DPL_EXPORT consteval operation_category operator|(
    operation_category lhs, operation_category rhs) noexcept {
    return static_cast<operation_category>(
        __DPL to_underlying(lhs) | __DPL to_underlying(rhs));
}

DPL_EXPORT consteval operation_category operator&(
    operation_category lhs, operation_category rhs) noexcept {
    return static_cast<operation_category>(
        __DPL to_underlying(lhs) & __DPL to_underlying(rhs));
}

DPL_EXPORT consteval bool operator==(
    operation_category lhs, operation_category rhs) noexcept {
    return __DPL to_underlying(lhs) == __DPL to_underlying(rhs);
}

DPL_EXPORT consteval bool operator!=(
    operation_category lhs, operation_category rhs) noexcept {
    return !(lhs == rhs);
}

} // namespace datapar

DPL_DEFAULT_NAMESPACE_END
