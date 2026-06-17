// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/simd_type.h"
#  include "dpl/core/type_traits/simd_native_type.h"
#  include "dpl/std/utility/forward.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <simd_type T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr auto DPL_VECTORCALL to_native_type(T&& src) noexcept {
    return static_cast<simd_native_type_t<T>>(__DPL forward<T>(src));
}

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
