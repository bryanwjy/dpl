// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/std/concepts/enumeration.h"
#  include "dpl/std/type_traits/underlying_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
template <simd_class T>
requires enumeration<simd_lane_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto to_underlying(T simd) noexcept {
    using To = underlying_type_t<simd_lane_type_t<T>>;
    return datapar::reinterpret<To>(simd);
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
