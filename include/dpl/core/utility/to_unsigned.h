// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/operations/reinterpret.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/type_traits/make_unsigned.h"
#endif

__DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
template <simd_type T>
requires integral<simd_element_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto to_unsigned(T simd) noexcept {
    using To = make_unsigned_t<simd_element_type_t<T>>;
    return datapar::reinterpret<To>(simd);
}
} // namespace datapar
__DPL_DEFAULT_NAMESPACE_END
