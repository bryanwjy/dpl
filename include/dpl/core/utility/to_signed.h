// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/basic/reinterpret.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/std/concepts/integral.h"
#  include "dpl/std/type_traits/make_signed.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {
template <simd_class T>
requires integral<simd_element_type_t<T>>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr auto to_unsigned(T simd) noexcept {
    using To = make_signed_t<simd_element_type_t<T>>;
    return datapar::reinterpret<To>(simd);
}
} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
