// Copyright 2025-2026 Bryan Wong
#pragma once

#include "dpl/config.h"

#if !DPL_MODULES
#  include "dpl/core/concepts/basic_type.h"
#  include "dpl/core/concepts/simd_class.h"
#  include "dpl/core/type_traits/basic_type.h"
#endif

DPL_DEFAULT_NAMESPACE_BEGIN
namespace datapar {

template <basic_simd_class T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr T to_basic_type(T src) noexcept {
    return src;
}

template <simd_class T>
DPL_ATTRIBUTES(_HIDE_FROM_ABI, ALWAYS_INLINE, NODISCARD)
constexpr basic_type_t<T> to_basic_type(T src) noexcept {
    if constexpr (requires { static_cast<basic_type_t<T>>(src); }) {
        return static_cast<basic_type_t<T>>(src);
    } else {
        return +src;
    }
}

} // namespace datapar
DPL_DEFAULT_NAMESPACE_END
