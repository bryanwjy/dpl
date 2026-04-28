// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#include "dpl/core/basic/to_native_type.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/type_traits/basic_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

DPL_EXPORT template <simd_class T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T to_basic_type(T src) noexcept {
    return src;
}

DPL_EXPORT template <simd_class T>
requires (!basic_simd_class<T>)
DPL_ATTRIBUTES(_HIDE_FROM_ABI, NODISCARD)
constexpr basic_type_t<T> DPL_VECTORCALL to_basic_type(T src) noexcept {
    if constexpr (explicitly_convertible_to<T, basic_type_t<T>>) {
        return static_cast<basic_type_t<T>>(src);
    } else {
        return datapar::to_native_type(src);
    }
}

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
